import utils
import tyro
import wandb

from transformers import AutoTokenizer

from rwkv_fast.modeling_rwkv import RwkvForCausalLM
import buffer_generator_full

from datasets import load_dataset

import schedulefree
import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm

from math import ceil

from string_env import MAX_NUM_PLAYERS


args = tyro.cli(utils.Args)

# args.sl_coef = 1
# args.sl_lr = 0
args.use_rl = True
args.use_bc = False
# args.use_sl = True
# args.use_wm = True


RWKV_NAME = utils.RWKV_NAMES[args.rwkv_id]

full_name = f"Imposter_{args.num_imposters + args.num_crewmates}_{RWKV_NAME.split('-')[-1]}_{args.seed}_{args.name}"

if args.track:
    wandb.init(
        project="amogus_sp2",
        name=full_name,
        config=vars(args)
    )

utils.random_seed(args.seed)

args.checkpoint_name = RWKV_NAME

if len(args.i_checkpoint_name) == 0:
    args.i_checkpoint_name = RWKV_NAME
        
if len(args.c_checkpoint_name) == 0:
    args.c_checkpoint_name = RWKV_NAME

if len(args.o_checkpoint_name) == 0:
    args.o_checkpoint_name = RWKV_NAME

tokenizer = AutoTokenizer.from_pretrained(utils.RWKV_NAMES[1], trust_remote_code=True)

i_model = RwkvForCausalLM.from_pretrained(args.i_checkpoint_name, torch_dtype=torch.bfloat16, device_map='cuda')
c_model = RwkvForCausalLM.from_pretrained(args.c_checkpoint_name, torch_dtype=torch.bfloat16, device_map='cuda')
o_model = RwkvForCausalLM.from_pretrained(args.o_checkpoint_name, torch_dtype=torch.bfloat16, device_map='cuda')

# print(c_model.head.weight.shape)
# nn.init.xavier_uniform_(c_model.head.weight[:MAX_NUM_PLAYERS, :])

i_optimizer = schedulefree.AdamWScheduleFree(i_model.parameters(), lr=args.learning_rate, eps=1e-5)

dataset = iter(load_dataset("JeanKaddour/minipile", streaming=True, split='train').shuffle(seed=42))


for iteration in tqdm(range(args.num_iterations), desc=" outer", position=0):
    with torch.no_grad():
        # bufs, sparse_i_wins, sparse_c_wins, discussion_benefits, held_out_benefits = buffer_generator_full.collect_real_buffers([i_model, c_model], [args.num_imposters + 1, args.num_crewmates - 1], tokenizer, num_worlds=args.num_worlds, num_players=args.num_imposters + args.num_crewmates, world_width=args.world_width, discussion_turns=args.discussion_turns, num_observers=0, discussion_reward_weight=args.discussion_reward_weight, reporting_alignment=50)
        bufs, sparse_i_wins, sparse_c_wins, discussion_benefits, held_out_benefits = buffer_generator_full.collect_real_buffers([i_model, c_model, o_model], [args.num_imposters, args.num_crewmates-1, 1], tokenizer, num_worlds=args.num_worlds, num_players=args.num_imposters + args.num_crewmates, world_width=args.world_width, discussion_turns=args.discussion_turns, num_observers=0, discussion_reward_weight=args.discussion_reward_weight, reporting_alignment=50, randomize=args.randomize)
        ibuf = bufs[0]
        cbuf = bufs[1]
        cur_i_rate = sum(sparse_i_wins) / len(sparse_i_wins)
        cur_c_rate = sum(sparse_c_wins) / len(sparse_c_wins)
        net_discussion_benefits = np.mean(discussion_benefits)
        print(discussion_benefits)
        print("*" * 20 + "IMPOSTER" + "*"*20)
        print(tokenizer.decode(ibuf.all_tokens[0]))
        print("*" * 48)
        print("imposters", cur_i_rate, "crewmates", cur_c_rate, "net discussion", net_discussion_benefits)

        for epoch in range(args.update_epochs):
            utils.reevaluate_logprobs(args, cbuf, c_model, epoch, args.update_epochs)

    track_dict = {}
    track_dict['cur_i_rate'] = cur_i_rate
    track_dict['cur_c_rate'] = cur_c_rate
    track_dict['net_discussion_benefits'] = net_discussion_benefits
    track_dict['reward'] = torch.sum(cbuf.rewards, dim=-1).mean()
    track_dict['imposter_reward'] = torch.sum(ibuf.rewards, dim=-1).mean()
    if args.track:
        wandb.log(track_dict)
        

    correct_advantages = cbuf.advantages[cbuf.token_flags > 1]
    advantages_mean = correct_advantages.mean()
    advantages_std = correct_advantages.std()

    icorrect_advantages = ibuf.advantages[ibuf.token_flags > 1]
    iadvantages_mean = icorrect_advantages.mean()
    iadvantages_std = icorrect_advantages.std()

    for _ in range(args.re_iterations):
        for epoch in range(args.update_epochs // 3):
            all_losses = {'bc_loss': 0, 'loss_ans': 0, 'loss_lm': 0, 'pg_loss': 0, 'entropy_loss': 0, 'v_loss': 0, 'kl_logratio': 0, 'sl_kl_logratio': 0, 'all_probs': [], 'minipile': 0}
            cur_losses = utils.do_train(args, ibuf, i_model, iadvantages_mean, iadvantages_std, epoch, False, args.update_epochs //  3, i_optimizer)
            for l in cur_losses:
                all_losses[l] += cur_losses[l]
            all_losses['minipile'] += utils.do_pile_training(args, i_model, dataset, tokenizer, i_optimizer)

            track_dict = {l: all_losses[l] for l in all_losses if not isinstance(all_losses[l], list)}
            if len(all_losses['all_probs']) != 0:
                track_dict['accuracy'] = np.mean(all_losses['all_probs'])
                track_dict['acc_std'] = np.std(all_losses['all_probs'])
                track_dict['min_acc'] = min(all_losses['all_probs'])
            if args.track:
                del track_dict['bc_loss']
                wandb.log(track_dict)

    if (iteration % 5 == 4 or iteration == args.num_iterations - 1) and args.track and args.save:
        saved_model_name = "trained_models/" + full_name + "/" + str(iteration)
        i_model.train()
        i_model.rwkv._rescale_layers()
        i_model.save_pretrained(saved_model_name)
        i_model.eval()
        i_model.rwkv._rescale_layers()
        
        utils.validate_load_success(saved_model_name, i_model)
if args.track:
    wandb.finish()
