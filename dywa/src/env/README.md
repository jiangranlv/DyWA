
## Directory Structure

```
env/
├─robot/
├─env/
├─task/
├─scene/
├─push_env.py
├─arm_env.py
├─util.py
├─common.py
└─.gitignore
```

* [robot](./robot) Robot classes (franka, UR5, ...)
* [env](./env) Environment-related modules
* [task](./task) Task-related modules (push, ...)
* [scene](./scene) Scene-related modules (tabletop, how many objects, ...)
* [push_env.py](./push_env.py): Base implementation for push-type environments.
* [arm_env.py](./arm_env.py): Main wrapper for robot-arm based push env.
* [util.py](./util.py): Utility functions for dealing with isaac gym.
* [common.py](./common.py): Common functions used across multiple environments.
