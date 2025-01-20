#!/bin/bash

S=0

C=trained_models/Pretrain_5_1b5_${S}_0_no_discuss/14

A=trained_models/Crewmate_5_1b5_${S}_1_full_no_discuss_14/39
B=trained_models/Crewmate_5_1b5_${S}_1_rl_sl_no_discuss_14_coef3/39
D=trained_models/Crewmate_5_1b5_${S}_1_rl_only_no_discuss_14_coef/39

python validate_all.py --num_worlds 30 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name $C --c_checkpoint_name $A --seed 100
python validate_all.py --num_worlds 30 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name $C --c_checkpoint_name $B --seed 100
python validate_all.py --num_worlds 30 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name $C --c_checkpoint_name $C --seed 100
python validate_all.py --num_worlds 30 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name $C --c_checkpoint_name $D --seed 100
