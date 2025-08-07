#!/bin/bash
cd dywa/exp/train

name='film_mlp'
root="/home/user/DyWA/output/dywa"

if [ ! -d "${root}/${name}" ]; then
mkdir -p "${root}/${name}"
fi

# Usage: bash run_teacher_stage1.sh

PYTORCH_JIT=0 python3 train_ppo_arm.py \
  +platform=debug \
  +env=icra_base \
  +run=icra_ours \
  ++env.seed=56081 \
  ++tag=policy \
  ++global_device=cuda:0 \
  ++path.root=/home/user/DyWA/output/dywa/teacher-stage1/ \
  ++icp_obs.icp.ckpt="imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920"