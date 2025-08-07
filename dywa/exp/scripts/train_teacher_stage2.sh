#!/bin/bash
cd dywa/exp/train

name='film_mlp'
root="/home/user/DyWA/output/dywa"

if [ ! -d "${root}/${name}" ]; then
mkdir -p "${root}/${name}"
fi

# Usage: bash run_teacher_stage2.sh [POLICY_CKPT_PATH]

POLICY_CKPT=${1:-"/home/user/DyWA/output/dywa/teacher-stage1/run-000/ckpt/"}

PYTORCH_JIT=0 python3 train_ppo_arm.py \
  +platform=debug \
  +env=icra_base \
  +run=icra_ours \
  ++env.seed=56081 \
  ++tag=student \
  ++global_device=cuda:0 \
  ++path.root=/home/user/DyWA/output/dywa/teacher-stage2/ \
  ++env.num_env=8192 \
  ++is_phase2=true \
  ++phase2.min_reset_to_update=65536 \
  ++agent.train.lr=2e-6 \
  ++agent.train.alr.initial_scale=6.67e-3 \
  ++icp_obs.icp.ckpt="imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920" \
  ++load_ckpt="${POLICY_CKPT}"