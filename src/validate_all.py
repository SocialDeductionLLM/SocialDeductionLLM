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

# factors: world height, world width, num tasks, num crewmates

all_configs = [
    "Environment",
    (2, 1, 4, 4),
    (1, 3, 4, 4),
    (2, 2, 4, 4),
    (2, 3, 4, 4),
    (3, 2, 4, 4),
    "Tasks",
    (2, 2, 2, 4),
    (2, 2, 3, 4),
    (2, 2, 4, 4),
    (2, 2, 5, 4),
    (2, 2, 6, 4),
    "Players",
    (2, 2, 4, 3),
    (2, 2, 4, 4),
    (2, 2, 4, 5),
]


args = tyro.cli(utils.Args)

RWKV_NAME = utils.RWKV_NAMES[args.rwkv_id]

utils.random_seed(args.seed)

args.checkpoint_name = RWKV_NAME

tokenizer = AutoTokenizer.from_pretrained(utils.RWKV_NAMES[1], trust_remote_code=True)

if len(args.i_checkpoint_name) == 0:
    args.i_checkpoint_name = RWKV_NAME
        
if len(args.c_checkpoint_name) == 0:
    args.c_checkpoint_name = RWKV_NAME

if len(args.o_checkpoint_name) == 0:
    args.o_checkpoint_name = args.i_checkpoint_name
    
i_model = RwkvForCausalLM.from_pretrained(args.i_checkpoint_name, torch_dtype=torch.bfloat16, device_map='cuda')
c_model = RwkvForCausalLM.from_pretrained(args.c_checkpoint_name, torch_dtype=torch.bfloat16, device_map='cuda')
o_model = RwkvForCausalLM.from_pretrained(args.o_checkpoint_name, torch_dtype=torch.bfloat16, device_map='cuda')

# c_model = buffer_generator_full.HybridModel(buffer_generator_full.DummyModel(is_imposter=False), c_model)

current_task = None
cache = {}
answers = {}

for config in all_configs:

    if isinstance(config, str):
        current_task = config
        continue

    torch.cuda.empty_cache()
    utils.random_seed(args.seed)

    if config in cache:
        cur_c_rate = cache[config]
    else:
        width, height, tasks, crewmates = config
        with torch.no_grad():
            _, sparse_i_wins, sparse_c_wins, discussion_benefits, held_out_benefits = buffer_generator_full.collect_real_buffers([i_model, c_model, o_model], [args.num_imposters, crewmates - 1, 1], tokenizer, num_worlds=args.num_worlds, num_players=args.num_imposters + crewmates, world_width=width, world_height=height, discussion_turns=args.discussion_turns, num_observers=0, reporting_alignment=50, num_tasks=tasks, do_calc=False)
            # ibuf = bufs[0]
            # cbuf = bufs[1]
            # print("*" * 20 + "IMPOSTER" + "*"*20)
            # print(tokenizer.decode(ibuf.all_tokens[0]))
            # print("*" * 48)
            cur_i_rate = sum(sparse_i_wins) / len(sparse_i_wins)
            cur_c_rate = sum(sparse_c_wins) / len(sparse_c_wins)
            net_discussion_benefits = np.mean(discussion_benefits)
            net_held_benefits = np.mean(held_out_benefits)
            # print("imposter win rate", cur_i_rate, "crewmate win rate", cur_c_rate)
            # print(sparse_wins)
            # print(np.mean(discussion_benefits))
            # del bufs

    
    print(current_task, config, cur_c_rate)
    cache[config] = cur_c_rate
    if current_task not in answers:
        answers[current_task] = []
    answers[current_task].append(cur_c_rate)
        

with open(f"logs/{args.name}.txt", "w") as text_file:
    for task, scores in answers.items():
        str_ans = f"{task:20}:  np.array({scores})\n"
        text_file.write(str_ans)
