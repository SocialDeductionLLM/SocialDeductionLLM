#!/bin/bash

S=0

C=trained_models/Pretrain_5_1b5_${S}_0_no_discuss/14

Ac=trained_models/Crewmate_5_1b5_${S}_1_full_no_discuss_14/39
Ai=trained_models/Imposter_5_1b5_${S}_1_sp/39

Bc=trained_models/Crewmate_5_1b5_${S}_2_sp/9
Bi=trained_models/Imposter_5_1b5_${S}_2_sp/9

Cc=trained_models/Crewmate_5_1b5_${S}_3_sp/9
Ci=trained_models/Imposter_5_1b5_${S}_3_sp/9

python validate_base.py --num_worlds 100 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name $C --c_checkpoint_name $Ac --o_checkpoint_name $C --seed 100 --name max_0_${S}

python validate_base.py --num_worlds 100 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name $Ai --c_checkpoint_name $C --o_checkpoint_name $C --seed 100 --name min_0_${S}




python validate_base.py --num_worlds 100 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name $Ai --c_checkpoint_name $Bc --o_checkpoint_name $C --seed 100 --name max_1_${S}

python validate_base.py --num_worlds 100 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name $Bi --c_checkpoint_name $Ac --o_checkpoint_name $C --seed 100 --name min_1_${S}




python validate_base.py --num_worlds 100 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name $Bi --c_checkpoint_name $Cc --o_checkpoint_name $C --seed 100 --name max_2_${S}

python validate_base.py --num_worlds 100 --num_iterations 1 --discussion_turns 8 --i_checkpoint_name $Ci --c_checkpoint_name $Bc --o_checkpoint_name $C --seed 100 --name min_2_${S}
