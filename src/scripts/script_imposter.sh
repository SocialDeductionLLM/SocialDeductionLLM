#!/bin/bash

# Crewmate iter 0 -> Crewmate_5_1b5_1_full/24

# Imposter iter 0 -> Imposter_5_1b5_1_sp/24
python train_imposter.py --rwkv_id 2 --num_worlds 30 --update_epochs 9 --num_iterations 25 --fl_gamma 3 --discussion_turns 8 --name "sp" --train_crewmates --no-train_imposters --no-clip_vloss --ema 0 --i_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --c_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --o_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --randomize --no-norm_adv  --seed 1 --learning_rate 3e-4 --sl_coef 3.0 --vf_coef 1.0 --discussion_reward_weight 1.0 --d_coef 0.2 --a_coef 1.0 --ent_coef 0.01 --kl_coef 0.05

# Crewmate iter 1 -> Crewmate_5_1b5_2_sp/9
python train_crewmate_real.py --rwkv_id 2 --num_worlds 30 --update_epochs 9 --num_iterations 10 --fl_gamma 3 --discussion_turns 8 --name "sp" --train_crewmates --no-train_imposters --no-clip_vloss --ema 0 --i_checkpoint_name trained_models/Imposter_5_1b5_1_sp/24 --c_checkpoint_name trained_models/Crewmate_5_1b5_1_full/24 --o_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --randomize --no-norm_adv  --seed 2 --learning_rate 3e-4 --sl_coef 3.0 --vf_coef 1.0 --discussion_reward_weight 1.0 --d_coef 0.2 --a_coef 1.0 --ent_coef 0.01 --kl_coef 0.05

# Imposter iter 1 -> Imposter_5_1b5_2_sp/9
python train_imposter.py --rwkv_id 2 --num_worlds 30 --update_epochs 9 --num_iterations 10 --fl_gamma 3 --discussion_turns 8 --name "sp" --train_crewmates --no-train_imposters --no-clip_vloss --ema 0 --i_checkpoint_name trained_models/Imposter_5_1b5_1_sp/24 --c_checkpoint_name trained_models/Crewmate_5_1b5_1_full/24 --o_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --randomize --no-norm_adv  --seed 2 --learning_rate 3e-4 --sl_coef 3.0 --vf_coef 1.0 --discussion_reward_weight 1.0 --d_coef 0.2 --a_coef 1.0 --ent_coef 0.01 --kl_coef 0.05


# Crewmate iter 2 -> Crewmate_5_1b5_3_sp/9
python train_crewmate_real.py --rwkv_id 2 --num_worlds 30 --update_epochs 9 --num_iterations 10 --fl_gamma 3 --discussion_turns 8 --name "sp" --train_crewmates --no-train_imposters --no-clip_vloss --ema 0 --i_checkpoint_name trained_models/Imposter_5_1b5_2_sp/9 --c_checkpoint_name trained_models/Crewmate_5_1b5_2_sp/9 --o_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --randomize --no-norm_adv  --seed 3 --learning_rate 3e-4 --sl_coef 3.0 --vf_coef 1.0 --discussion_reward_weight 1.0 --d_coef 0.2 --a_coef 1.0 --ent_coef 0.01 --kl_coef 0.05

# Imposter iter 2 -> Imposter_5_1b5_3_sp/9
python train_imposter.py --rwkv_id 2 --num_worlds 30 --update_epochs 9 --num_iterations 10 --fl_gamma 3 --discussion_turns 8 --name "sp" --train_crewmates --no-train_imposters --no-clip_vloss --ema 0 --i_checkpoint_name trained_models/Imposter_5_1b5_2_sp/9 --c_checkpoint_name trained_models/Crewmate_5_1b5_2_sp/9 --o_checkpoint_name trained_models/Pretrain_5_1b5_0_no_discuss/14 --randomize --no-norm_adv  --seed 3 --learning_rate 3e-4 --sl_coef 3.0 --vf_coef 1.0 --discussion_reward_weight 1.0 --d_coef 0.2 --a_coef 1.0 --ent_coef 0.01 --kl_coef 0.05
