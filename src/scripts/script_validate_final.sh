#!/bin/bash
C=trained_models/Crewmate_5_1b5_1_rl_scratch/24
# C=trained_models/Crewmate_5_1b5_1_rl_sl/24
# C=trained_models/Crewmate_5_1b5_1_full/24
# C=trained_models/Pretrain_5_1b5_0_no_discuss/14
# base for Listening Only

echo "BASE"

python validate.py --rwkv_id 2 --num_worlds 100 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --c_checkpoint_name $C --seed 100 --world_height 2 --world_width 2 --num_tasks 4 --num_crewmates 4












python validate.py --rwkv_id 2 --num_worlds 30 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --c_checkpoint_name $C --seed 100 --world_height 2 --world_width 1 --num_tasks 4 --num_crewmates 4

python validate.py --rwkv_id 2 --num_worlds 30 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --c_checkpoint_name $C --seed 100 --world_height 1 --world_width 3 --num_tasks 4 --num_crewmates 4

python validate.py --rwkv_id 2 --num_worlds 30 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --c_checkpoint_name $C --seed 100 --world_height 2 --world_width 2 --num_tasks 4 --num_crewmates 4

python validate.py --rwkv_id 2 --num_worlds 30 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --c_checkpoint_name $C --seed 100 --world_height 2 --world_width 3 --num_tasks 4 --num_crewmates 4

python validate.py --rwkv_id 2 --num_worlds 30 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --c_checkpoint_name $C --seed 100 --world_height 3 --world_width 2 --num_tasks 4 --num_crewmates 4




echo "NOW DOING TASKS"


python validate.py --rwkv_id 2 --num_worlds 30 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --c_checkpoint_name $C --seed 100 --world_height 2 --world_width 2 --num_tasks 2 --num_crewmates 4

python validate.py --rwkv_id 2 --num_worlds 30 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --c_checkpoint_name $C --seed 100 --world_height 2 --world_width 2 --num_tasks 3 --num_crewmates 4

# python validate.py --rwkv_id 2 --num_worlds 30 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --c_checkpoint_name $C --seed 100 --world_height 2 --world_height 2 --num_tasks 4 --num_crewmates 4
echo "REPEAT"

python validate.py --rwkv_id 2 --num_worlds 30 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --c_checkpoint_name $C --seed 100 --world_height 2 --world_width 2 --num_tasks 5 --num_crewmates 4

python validate.py --rwkv_id 2 --num_worlds 30 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --c_checkpoint_name $C --seed 100 --world_height 2 --world_width 2 --num_tasks 6 --num_crewmates 4


echo "NOW DOING NUM PLAYERS"


python validate.py --rwkv_id 2 --num_worlds 30 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --c_checkpoint_name $C --seed 100 --world_height 2 --world_width 2 --num_tasks 4 --num_crewmates 3

python validate.py --rwkv_id 2 --num_worlds 30 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --c_checkpoint_name $C --seed 100 --world_height 2 --world_width 2 --num_tasks 4 --num_crewmates 5
