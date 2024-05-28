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

import copy

from string_env import MAX_NUM_PLAYERS


args = tyro.cli(utils.Args)

# args.sl_coef = 1
# args.sl_lr = 0
args.use_rl = False
args.use_bc = True
# args.use_sl = True
args.use_wm = True


RWKV_NAME = utils.RWKV_NAMES[args.rwkv_id]

full_name = f"Pretrain_{args.num_imposters + args.num_crewmates}_{RWKV_NAME.split('-')[-1]}_{args.seed}_{args.name}"

if args.track:
    wandb.init(
        project="amogus_pretrain2",
        name=full_name,
        config=vars(args)
    )

utils.random_seed(args.seed)

args.checkpoint_name = RWKV_NAME

tokenizer = AutoTokenizer.from_pretrained(utils.RWKV_NAMES[1], trust_remote_code=True)

base_model = RwkvForCausalLM.from_pretrained(args.checkpoint_name, torch_dtype=torch.bfloat16, device_map='cuda')
c_model = RwkvForCausalLM.from_pretrained(args.checkpoint_name, torch_dtype=torch.bfloat16, device_map='cuda')
i_model = c_model

nn.init.xavier_uniform_(base_model.head.weight[:MAX_NUM_PLAYERS, :])
nn.init.xavier_uniform_(c_model.head.weight[:MAX_NUM_PLAYERS, :])

c_optimizer = schedulefree.AdamWScheduleFree(c_model.parameters(), lr=args.learning_rate, eps=1e-5)
i_optimizer = c_optimizer


dummy_model_crewmate = buffer_generator_full.HybridModel(buffer_generator_full.DummyModel(is_imposter=False), base_model)
dummy_model_imposter = buffer_generator_full.HybridModel(buffer_generator_full.DummyModel(is_imposter=True), base_model)

dataset = iter(load_dataset("JeanKaddour/minipile", streaming=True, split='train').shuffle(seed=42))


for iteration in tqdm(range(args.num_iterations), desc=" outer", position=0):
    # ho_model = copy.deepcopy(c_model)
    with torch.no_grad():
        bufs, sparse_i_wins, sparse_c_wins, discussion_benefits, held_out_benefits = buffer_generator_full.collect_real_buffers([dummy_model_imposter, dummy_model_crewmate], [args.num_imposters, args.num_crewmates], tokenizer, num_worlds=args.num_worlds, num_players=args.num_imposters + args.num_crewmates, world_width=args.world_width, discussion_turns=args.discussion_turns, num_observers=0, discussion_reward_weight=args.discussion_reward_weight, reporting_alignment=50, randomize=args.randomize)
        ibuf = bufs[0]
        cbuf = bufs[1]
        print("*" * 20 + "IMPOSTER" + "*"*20)
        print(tokenizer.decode(ibuf.all_tokens[0]))
        print("*" * 48)

    correct_advantages = cbuf.advantages[cbuf.token_flags > 1]
    advantages_mean = correct_advantages.mean()
    advantages_std = correct_advantages.std()

    icorrect_advantages = ibuf.advantages[ibuf.token_flags > 1]
    iadvantages_mean = icorrect_advantages.mean()
    iadvantages_std = icorrect_advantages.std()

    # print(cbuf.returns, ibuf.returns)

    for _ in range(args.re_iterations):
        for epoch in range(args.update_epochs):
            all_losses = {'bc_loss': 0, 'loss_ans': 0, 'loss_lm': 0, 'pg_loss': 0, 'entropy_loss': 0, 'v_loss': 0, 'kl_logratio': 0, 'sl_kl_logratio': 0, 'all_probs': [], 'minipile': 0}
            cur_losses = utils.do_train(args, cbuf, c_model, advantages_mean, advantages_std, epoch, True, args.update_epochs, c_optimizer, ho_model=base_model)
            for l in cur_losses:
                all_losses[l] += cur_losses[l]

            cur_losses = utils.do_train(args, ibuf, i_model, iadvantages_mean, iadvantages_std, epoch // args.num_crewmates, False, ceil(args.update_epochs /  args.num_crewmates), i_optimizer)
            for l in cur_losses:
                all_losses[l] += cur_losses[l]
            all_losses['minipile'] += utils.do_pile_training(args, i_model, dataset, tokenizer, i_optimizer)

            track_dict = {l: all_losses[l] for l in all_losses if not isinstance(all_losses[l], list)}
            if len(all_losses['all_probs']) != 0:
                track_dict['accuracy'] = np.mean(all_losses['all_probs'])
                track_dict['acc_std'] = np.std(all_losses['all_probs'])
                track_dict['min_acc'] = min(all_losses['all_probs'])
            if args.track:
                wandb.log(track_dict)

    if (iteration % 5 == 4 or iteration == args.num_iterations - 1) and args.track:
        saved_model_name = "trained_models/" + full_name + "/" + str(iteration)
        c_model.train()
        c_model.rwkv._rescale_layers()
        c_model.save_pretrained(saved_model_name)
        c_model.eval()
        c_model.rwkv._rescale_layers()
        
        utils.validate_load_success(saved_model_name, c_model)
        
if args.track:
    wandb.finish()
