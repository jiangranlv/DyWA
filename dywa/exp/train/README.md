# Training Guideline for DyWA

## 1. Loading and Running a Pretrained Policy

To run our pretrained policy in a simulation, download the pretrained weights from [here](https://huggingface.co/Steve3zz/Dywa_abs_1view/tree/main), and place the `test_set.json` file under `/input/DGN/`.

Then run the following command to evaluate on unseen objects during training (make sure to replace the `load_student` parameter in `dywa/exp/train/eval_student_unseen_obj.sh` with the path to the pretrained weights you just downloaded):

```bash
bash dywa/exp/train/eval_student_unseen_obj.sh
```

**Note**: You may have to setup display following [Instruction](#extra-tips) for visualization.

In case you'd like to dive deeper, we also include documentation about training the policy yourself in the below sections.

## 2. RL Teacher Policy Training

If you'd like to train the policy yourself, you may run the training script as follows:

```bash
bash train_teacher_stage1.sh
```

If the training run results in an error, you may need to run:
```bash
sudo chmod -R 777 /home/user/.cache/
```

If you encounter network problems with Hugging Face, consider using a mirror site:
```bash
export HF_ENDPOINT='https://hf-mirror.com'
```

By default, the results of the training will be stored in `/home/user/DyWA/output/dywa/teacher-stage1/run-{:03d}`.

In general, we disable JIT for the purposes of the current code release, as its stability depends on your hardware.
In the case that your particular GPU+docker setup supports JIT compilation, you may enable `torch.jit.script()` as follows:
```bash
PYTORCH_JIT=1 python3 ...
```

## 3. RL Teacher Policy Evaluation

```bash
PYTORCH_JIT=0 python3 show_ppo_arm.py +platform=debug +env=icra_base +run=icra_ours ++env.seed=56081 ++tag=policy ++global_device=cuda:0 ++path.root=/home/user/DyWA/output/dywa/ ++icp_obs.icp.ckpt="imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920" ++load_ckpt="${POLICY_CKPT}" ++env.num_env=1
```
Replace `${POLICY_CKPT}` with the name of the policy that you have trained or downloaded.

By default, we disable JIT for the purposes of the current code release.
In the case that your particular GPU+docker setup supports JIT compilation, you may enable `torch.jit.script()` as follows:
```bash
PYTORCH_JIT=1 python3 ...
```

Assuming all goes well, you should see an image like this:

![policy-image](../../../fig/policy.png)

To enable graphics rendering, see [extra tips](#extra-tips).


## 4. Teacher-Student Distillation

To distill the privileged teacher to the student, first run the second phase of policy fine-tuning to reduce the action-space:

```bash
bash train_teacher_stage2.sh /path/to/policy_ckpt_stage1.pt
```

Afterward, replace the default `icp_obs.ckpt` and `load_ckpt` in `/home/user/DyWA/dywa/src/data/cfg/run/teacher_base.yaml` with the ones you obtained from your own training or you can download weights we train form [here](https://huggingface.co/Steve3zz/Dywa_abs_1view/tree/main), then run the distillation script as follows:

Unkonw state 1 view setting:
```bash
bash dywa/exp/scripts/train_distill_abs_goal_1view.sh
```

Unkown state 3 view setting:
```bash
bash dywa/exp/scripts/train_distill_abs_goal_3view.sh
```

Know State 3 view setting:
```bash
bash dywa/exp/scripts/train_distill_rel_goal_3view.sh
```

## Extra Tips

To enable the visualization during [policy training](#policy-training) or [policy evaluation](#policy-evaluation), you may need to add `++env.use_viewer=1` to the command line arguments.

Note that this will slow down the training process, so it's generally not recommended - only for debugging.

In case the visualization window does not start, ensure that the `${DISPLAY}` environment variable is configured to match that of the host system, which can be checked by running:
```bash
echo $DISPLAY
```
_outside_ of the docker container, i.e. in your _host_ system. Then, _inside_ the docker container:

```bash
export $DISPLAY=...
```
so that it matches the output of the earlier command (`$echo ${DISPLAY}`) in your host system.

### Experiment Tracking & API Setup

While the models and training progressions are also stored locally, You may track our model progress via [WanDB](https://wandb.ai/) and store the pretrained models with [HuggingFace](https://huggingface.co/).
To configure both APIs, run `wandb login` / `huggingface-cli login`.

Afterward, you can replace the `+platform=debug` directive in the command line with `+platform=dr`.

### Troubleshooting

In the case that a prior `jit` compilation has failed, it may be necessary to clear the jit cache:

```
rm -rf ~/.cache/torch_extensions
```

### Details about CLI options

When running the training (`train_ppo_arm.py`) or evaluation (`show_ppo_arm.py`) scripts, we use a number of CLI options to configure the runs. We describe details about those options here:

* `+platform` configures the training run depending on your hardware; by default we use `debug` which disables WanDB logging.
* `+env` refers to the environment configuration such as the object set and domain randomization parameters. (by default, we load `icra_base` which can be inspected [here](../../src/data/cfg/env/icra_base.yaml) which matches the hyperparameters in the paper.)
* `+run` refers to the policy configuration (by default, we load `icra_ours` which can be inspected [here](../../src/data/cfg/run/icra_ours.yaml) which matches hyperparameters as in the paper.)
* `++env.seed` overrides the RNG seeds that controls domain randomization.
* `++env.num_env` overrides the number of environments to simulate in parallel.
* `++global_device`, formatted as `cuda:{INDEX}`, configures the GPU device on which the simulation and training will be run.
* `++path.root` configures the local directory to which the run logs and weights will be saved.
* `++icp_obs.icp.ckpt` configures the pretrained weights of the _representation model_. It may point to a local path or to a huggingface repository formatted as `entity/repository:filename`.
* `++load_ckpt` loads the pretrained weights of the _policy_. Like `icp_obs.icp.ckpt`, it may point to a local path or to a huggingface repository formatted as `entity/repository:filename`.
* `++tag` sets the run tag for remote logging in `wandb`.
