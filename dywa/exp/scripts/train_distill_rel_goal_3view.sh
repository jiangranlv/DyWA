#!/bin/bash
cd dywa/exp/train

name='rel_goal_3view'
root="/home/user/DyWA/output/dywa"

GPU=${1:-0}

if [ ! -d "${root}/${name}" ]; then
mkdir -p "${root}/${name}"
fi

PYTORCH_JIT=0 python3 train_rma.py \
+platform=debug \
+env=rel_goal_3view \
+run=teacher_base \
+student=dywa_id_sconv_his5_lr6e4_film_scale \
++name="$name" \
++path.root="${root}/${name}" \
++env.num_env=1024 \
++global_device=cuda:${GPU} \
++student.norm="ln" \
++add_teacher_state=1 \
++student.state_keys=\ \[\'rel_goal\',\ \'hand_state\',\ \'robot_state\',\ \'previous_action\'\] \
# &> "$root/$name/out.out"