#!/bin/bash

S=0

python pretrainer.py --rwkv_id 2 --num_worlds 30 --update_epochs 20 --learning_rate 3e-4 --num_iterations 15 --fl_gamma 0 --discussion_turns 0 --discussion_reward_weight 1.0 --ema 0 --no-clip_vloss --kl_coef 0.1 --sl_coef 3.0 --randomize --ent_coef 0.01 --name "no_discuss" --seed 0 --other_seed $S --use_wm

C=trained_models/Pretrain_5_1b5_${S}_0_no_discuss/14

python train_crewmate.py --rwkv_id 2 --num_worlds 30 --update_epochs 15 --num_iterations 40 --fl_gamma 0 --discussion_turns 8 --name "full_no_discuss_14" --train_crewmates --no-train_imposters --no-clip_vloss --ema 0 --i_checkpoint_name $C --c_checkpoint_name $C --o_checkpoint_name $C --randomize --no-norm_adv --seed 1 --learning_rate 3e-4 --sl_coef 3.0 --vf_coef 1.0 --discussion_reward_weight 1.0 --d_coef 0.2 --a_coef 1.0 --ent_coef 0.01 --kl_coef 0.05 --other_seed $S --use_wm

python train_crewmate.py --rwkv_id 2 --num_worlds 30 --update_epochs 15 --num_iterations 40 --fl_gamma 0 --discussion_turns 8 --name "rl_sl_no_discuss_14_coef3" --train_crewmates --no-train_imposters --no-clip_vloss --ema 0 --i_checkpoint_name $C --c_checkpoint_name $C --o_checkpoint_name $C --randomize --no-norm_adv --seed 1 --learning_rate 3e-4 --sl_coef 0.1 --vf_coef 1.0 --discussion_reward_weight 0.0 --d_coef 0.0 --a_coef 1.0 --ent_coef 0.01 --kl_coef 0.05 --other_seed $S --use_wm

python train_crewmate.py --rwkv_id 2 --num_worlds 30 --update_epochs 15 --num_iterations 40 --fl_gamma 0 --discussion_turns 8 --name "rl_only_no_discuss_14_coef" --train_crewmates --no-train_imposters --no-clip_vloss --ema 0 --i_checkpoint_name $C --c_checkpoint_name $C --o_checkpoint_name $C --randomize --no-norm_adv --seed 1 --learning_rate 3e-4 --sl_coef 0.0 --vf_coef 1.0 --discussion_reward_weight 0.0 --d_coef 0.2 --a_coef 1.0 --ent_coef 0.01 --kl_coef 0.05 --other_seed $S --use_wm
