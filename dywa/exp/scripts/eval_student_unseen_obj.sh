#!/bin/bash
cd dywa/exp/train

name='dywa'
root="/home/user/DyWA/output/test_rma"

GPU=${1:-0}

if [ ! -d "${root}/${name}" ]; then
mkdir -p "${root}/${name}"
fi

PYTORCH_JIT=0 python3 test_rma.py \
+platform=debug \
+env=abs_goal_1view \
+run=teacher_base \
+student=dywa/base \
++name="$name" \
++path.root="${root}/${name}" \
++env.num_env=60 \
++global_device=cuda:${GPU} \
++student.norm="ln" \
+load_student=/home/user/DyWA/output/dywa/film_mlp/dywa/ckpt/last.ckpt \
++plot_pc=0 \
++dagger_train_env.anneal_step=1 \
++add_teacher_state=1 \
++student.decoder.film_mlp=1 \
++env.single_object_scene.filter_file=/input/DGN/test_set.json \
++monitor.num_env_record=60 \
++env.single_object_scene.mode=valid \
++log_categorical_results=True \
# &> "$root/$name/out.out"