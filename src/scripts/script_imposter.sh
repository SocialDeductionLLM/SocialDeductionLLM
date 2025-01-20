#!/bin/bash

S=0

C=trained_models/Pretrain_5_1b5_${S}_0_no_discuss/14

# Crewmate iter 0 -> Crewmate_5_1b5_${S}_1_full/24
# Imposter iter 0 -> Imposter_5_1b5_${S}_1_sp/24
python train_imposter.py --rwkv_id 2 --num_worlds 30 --update_epochs 15 --num_iterations 40 --fl_gamma 0 --discussion_turns 8 --name "sp" --train_crewmates --no-train_imposters --no-clip_vloss --ema 0 --i_checkpoint_name $C --c_checkpoint_name $C --o_checkpoint_name $C --randomize --no-norm_adv  --seed 1 --learning_rate 3e-4 --sl_coef 3.0 --vf_coef 1.0 --discussion_reward_weight 1.0 --d_coef 0.2 --a_coef 1.0 --ent_coef 0.01 --kl_coef 0.05 --other_seed $S --use_wm


A=trained_models/Crewmate_5_1b5_${S}_1_full_no_discuss_14/39
B=trained_models/Imposter_5_1b5_${S}_1_sp/39

# Crewmate iter 1 -> Crewmate_5_1b5_2_sp/9
python train_crewmate_real.py --rwkv_id 2 --num_worlds 30 --update_epochs 15 --num_iterations 10 --fl_gamma 0 --discussion_turns 8 --name "sp" --train_crewmates --no-train_imposters --no-clip_vloss --ema 0 --i_checkpoint_name $B --c_checkpoint_name $A --o_checkpoint_name $C --randomize --no-norm_adv  --seed 2 --learning_rate 3e-4 --sl_coef 3.0 --vf_coef 1.0 --discussion_reward_weight 1.0 --d_coef 0.2 --a_coef 1.0 --ent_coef 0.01 --kl_coef 0.05 --other_seed $S --use_wm

# Imposter iter 1 -> Imposter_5_1b5_2_sp/9
python train_imposter.py --rwkv_id 2 --num_worlds 30 --update_epochs 15 --num_iterations 10 --fl_gamma 0 --discussion_turns 8 --name "sp" --train_crewmates --no-train_imposters --no-clip_vloss --ema 0 --i_checkpoint_name $B --c_checkpoint_name $A --o_checkpoint_name $C --randomize --no-norm_adv  --seed 2 --learning_rate 3e-4 --sl_coef 3.0 --vf_coef 1.0 --discussion_reward_weight 1.0 --d_coef 0.2 --a_coef 1.0 --ent_coef 0.01 --kl_coef 0.05 --other_seed $S --use_wm


A=trained_models/Crewmate_5_1b5_${S}_2_sp/9
B=trained_models/Imposter_5_1b5_${S}_2_sp/9

# Crewmate iter 2 -> Crewmate_5_1b5_3_sp/9
python train_crewmate_real.py --rwkv_id 2 --num_worlds 30 --update_epochs 15 --num_iterations 10 --fl_gamma 0 --discussion_turns 8 --name "sp" --train_crewmates --no-train_imposters --no-clip_vloss --ema 0 --i_checkpoint_name $B --c_checkpoint_name $A --o_checkpoint_name $C --randomize --no-norm_adv  --seed 3 --learning_rate 3e-4 --sl_coef 3.0 --vf_coef 1.0 --discussion_reward_weight 1.0 --d_coef 0.2 --a_coef 1.0 --ent_coef 0.01 --kl_coef 0.05 --other_seed $S --use_wm

# Imposter iter 2 -> Imposter_5_1b5_3_sp/9
python train_imposter.py --rwkv_id 2 --num_worlds 30 --update_epochs 15 --num_iterations 10 --fl_gamma 0 --discussion_turns 8 --name "sp" --train_crewmates --no-train_imposters --no-clip_vloss --ema 0 --i_checkpoint_name $B --c_checkpoint_name $A --o_checkpoint_name $C --randomize --no-norm_adv  --seed 3 --learning_rate 3e-4 --sl_coef 3.0 --vf_coef 1.0 --discussion_reward_weight 1.0 --d_coef 0.2 --a_coef 1.0 --ent_coef 0.01 --kl_coef 0.05 --other_seed $S --use_wm
