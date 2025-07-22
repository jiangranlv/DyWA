<h1 align="center">🤖 <em>DyWA</em>: Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation</h1>



<br>
  <a href="https://jiangranlv.github.io/">Jiangran Lyu</a>,
  <a href="https://openreview.net/profile?id=~Ziming_Li2">Ziming Li</a>,
  <a href="https://scholar.google.com/citations?user=wRBbtl8AAAAJ&hl=en">Xuesong Shi</a>,
  <a href="https://co1one.github.io/">Chaoyi Xu</a>,
  <a href="https://cfcs.pku.edu.cn/people/faculty/yizhouwang/index.htm">Yizhou Wang</a><sup>&dagger;</sup>,
  <a href="https://hughw19.github.io/">He Wang</a><sup>&dagger;</sup>
<br>
<p>
  <sup>†</sup> Corresponding Author
</p>


1. CFCS, School of Computer Science, Peking University  
2. Galbot

Project website: [https://pku-epic.github.io/DyWA/](https://pku-epic.github.io/DyWA/)

---

## Table of Contents



- 1 [Workspace Setup](#1-setup)
  - 1.1 [Docker setup](#choice-1-docker-setup)
  - 1.2 [Package Setup](#choice-2-package-setup)
  - 1.3 [Assets Setup](#13-assets-setup)
- 2 [Policy Training](#2-policy-training)
- 3 [Policy Evaluation](#3-policy-evaluation)

## 1. Setup
> Note: We highly recommand users with docker setup.

### Choice 1: Docker Setup

Refer to the instructions in [docker](./docker)，For the PyTorch3D wheel built for CUDA 11.3 required by the DockerFile, do one of the following:
  1. Clone the [PyTorch3D repository](https://github.com/facebookresearch/pytorch3d) directory and build the wheel yourself.
  ```bash
  git clone --branch v0.7.2 https://github.com/facebookresearch/pytorch3d.git
  cd pytorch3d
  python setup.py bdist_wheel
  ```

  2. Download a pre-built pytorch3d==0.7.2 wheel for CUDA 11.3 in [here](https://huggingface.co/datasets/Steve3zz/pytorch3d_0.7.2_wheel_for_cuda_11.3/tree/main)
  
  place it under ./docker/

### Choice 2: Package Setup

#### Isaac Gym

First, download isaac gym from [here](https://developer.nvidia.com/isaac-gym) and extract them to the `${IG_PATH}` host directory
that you configured during docker setup. By default, we assume this is `/home/DyWA/isaacgym`, which maps to`/opt/isaacgym` directory inside the container.
In other words, the resulting directory structure should look like:

```bash
$ tree /opt/isaacgym -d -L 1

/opt/isaacgym
|-- assets
|-- docker
|-- docs
|-- licenses
`-- python
```

(If `tree` command is not found, you may simply install it via `sudo apt-get install tree`.)

Afterward, follow the instructions in the referenced page to install the isaac gym package.

Alternatively, assuming that the isaac gym package has been downloaded and extracted in the correct directory(`/opt/isaacgym`),
we provide the default setups for isaac gym installation in the [setup script](./setup.sh)
in the [following section](#python-package), which handles the installation automatically.

#### Python Package

Then, inside the docker image (assuming `${PWD}` is the repo root), run the [setup script](./setup.sh):

```bash
bash setup.sh
```


To test if the installation succeeded, you can run:
```bash
python3 -c 'import isaacgym; print("OK")'
python3 -c 'import dywa; print("OK")'
```

### 1.3 Assets Setup

We release a pre-processed version of the object mesh assets from [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) in [here](https://huggingface.co/imm-unicorn/corn-public/resolve/main/DGN.tar.gz).

After downloading the assets, extract them to `/path/to/data/DGN` in the _host_ container, so that `/path/to/data` matches the directory
configured in [docker/run.sh](docker/run.sh), i.e.

```bash
mkdir -p /path/to/data/DGN
tar -xzf DGN.tar.gz -C /path/to/data/DGN
```

so that the resulting directory structure _inside_ the docker container looks as follows:

```bash
$ tree /input/DGN --filelimit 16 -d     

/input/DGN
|-- coacd
`-- meta-v8
    |-- cloud
    |-- cloud-2048
    |-- code
    |-- hull
    |-- meta
    |-- new_pose
    |-- normal
    |-- normal-2048
    |-- pose
    |-- unique_dgn_poses
    `-- urdf
```

## 2. Policy Training

Navigate to the policy training directory in `dywa/exp/train` and follow the instructions in the [README](./dywa/exp/train/README.md).

## 3. Policy Evaluation

To run our pretrained policy in a simulation, download the pretrained weights from [here](https://huggingface.co/Steve3zz/Dywa_abs_1view/tree/main), and place the `test_set.json` file under `/input/DGN/`.

Then run the following command to evaluate on unseen objects during training (make sure to replace the `load_student` parameter in `dywa/exp/scripts/eval_student_unseen_obj.sh` with the path to the pretrained weights you just downloaded):

Unkown State Unseen Object Setting:
```bash
bash dywa/exp/scripts/eval_student_unseen_obj.sh
```

Evaluation results will be saved to `/home/user/DyWA/output/test_rma/`.

<img src="./fig/categorical_result.png" alt="Success rate on each kind of object" width="300"/>

For _visualizing_ the behavior of the policy, running too many environments in parallel may cause significant lag in your system.
Instead, adjust the number of parallel environment by changing `++env.num_env=${NUM_ENV}` and turn on the gui with `++env.use_viewer=1 ++draw_debug_lines=1`, and don't forget to export the `${DISPLAY}` variable to match the monitor settings from the host environment.

For detailed setup or troubleshooting, please refer to [README](./dywa/exp/train/README.md).


## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{lyu2025dywa,
  title={DyWA: Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation},
  author={Jiangran Lyu and Ziming Li and Xuesong Shi and Chaoyi Xu and Yizhou Wang and He Wang},
  booktitle={Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

## Acknowledgement
This work is built upon and further extended from the prior work [**CORN: Contact-based Object Representation for Nonprehensile Manipulation of General Unseen Objects**](https://sites.google.com/view/contact-non-prehensile). We sincerely thank the authors of CORN for their valuable contribution and for making their work publicly available.

## License

 This work and the dataset are licensed under [CC BY-NC 4.0][cc-by-nc].

 [![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

 [cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
 [cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png