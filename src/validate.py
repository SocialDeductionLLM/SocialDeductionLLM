import utils
import tyro
import wandb

from transformers import AutoTokenizer

from rwkv_fast.modeling_rwkv import RwkvForCausalLM
import buffer_generator_full

from datasets import load_dataset

import schedulefree
import torch

import numpy as np

from tqdm import tqdm

from math import ceil


args = tyro.cli(utils.Args)

RWKV_NAME = utils.RWKV_NAMES[args.rwkv_id]

utils.random_seed(args.seed)

args.checkpoint_name = RWKV_NAME

tokenizer = AutoTokenizer.from_pretrained(utils.RWKV_NAMES[1], trust_remote_code=True)

if len(args.i_checkpoint_name) == 0:
    args.i_checkpoint_name = RWKV_NAME
        
if len(args.c_checkpoint_name) == 0:
    args.c_checkpoint_name = RWKV_NAME
    
i_model = RwkvForCausalLM.from_pretrained(args.i_checkpoint_name, torch_dtype=torch.bfloat16, device_map='cuda')
c_model = RwkvForCausalLM.from_pretrained(args.c_checkpoint_name, torch_dtype=torch.bfloat16, device_map='cuda')

# c_model = buffer_generator_full.HybridModel(buffer_generator_full.DummyModel(is_imposter=False), c_model)

for iteration in tqdm(range(args.num_iterations), desc=" outer", position=0):
    with torch.no_grad():
        bufs, sparse_i_wins, sparse_c_wins, discussion_benefits, held_out_benefits = buffer_generator_full.collect_real_buffers([i_model, c_model], [args.num_imposters + 1, args.num_crewmates - 1], tokenizer, num_worlds=args.num_worlds, num_players=args.num_imposters + args.num_crewmates, world_width=args.world_width, world_height=args.world_height, discussion_turns=args.discussion_turns, num_observers=0, reporting_alignment=50, num_tasks=args.num_tasks)
        ibuf = bufs[0]
        cbuf = bufs[1]
        # print("*" * 20 + "IMPOSTER" + "*"*20)
        # print(tokenizer.decode(ibuf.all_tokens[0]))
        # print("*" * 48)
        cur_i_rate = sum(sparse_i_wins) / len(sparse_i_wins)
        cur_c_rate = sum(sparse_c_wins) / len(sparse_c_wins)
        net_discussion_benefits = np.mean(discussion_benefits)
        net_held_benefits = np.mean(held_out_benefits)
        print("imposter win rate", cur_i_rate, "crewmate win rate", cur_c_rate)
        # print(sparse_wins)
        print(np.mean(discussion_benefits))
        
