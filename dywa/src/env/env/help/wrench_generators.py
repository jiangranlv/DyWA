#!/usr/bin/env python3
"""
Mostly archived...
"""


class RandomWrenchGenerator:
    def __init__(self,
                 num_env: int,
                 device: th.device):

        self.num_env = num_env
        self.device = device

    def __call__(self, *args, **kwds):
        wrench = th.randn((self.num_env, 6),
                          dtype=th.float,
                          device=self.device)
        return wrench


class ConstantWrenchGenerator:
    def __init__(self,
                 num_env: int,
                 device: th.device):

        self.num_env = num_env
        self.device = device

    def __call__(self, *args, **kwds):
        wrench = th.zeros((self.num_env, 6),
                          dtype=th.float,
                          device=self.device)
        wrench[..., 1] = 1
        return wrench


class ManualWrenchGenerator:
    def __init__(self,
                 num_env: int, device: th.device):
        self.num_env = num_env
        self.device = device

    def __call__(self, obs: Dict[str, th.Tensor]):
        # FIXME: hardcoded DT !!
        DT: float = 1.0 / 240.0
        # K_VEL: float = 10.0 / DT / 1024
        # K_WRENCH: float = 10.0 / DT / 1000
        # FIXME: hardcoded mass !!
        MASS: float = 0.170
        # FIXME: hardcoded MAX_WRENCH !!
        MAX_WRENCH: float = +2.0
        MAX_ACCEL: float = MAX_WRENCH / MASS

        goal = obs['goal']
        s = obs['object_state']
        obj_pos = s[..., :3]
        obj_vel = s[..., 7:13]

        # initial state & residual
        disp = goal - obj_pos[..., :2]
        u = disp / th.linalg.norm(disp, dim=-1, keepdim=True)
        v = th.stack([-u[..., 1], u[..., 0]], dim=-1)
        R = th.stack([u, v], dim=-1)  # world to local

        x1 = th.zeros_like(s[..., 7:13])  # NX6
        x1[..., :2] = th.einsum('...ij, ...j -> ...i', R, disp)
        # v0 = th.einsum('...ij, ...j -> ...i', R, obj_vel)
        v0 = obj_vel
        v0[..., :2] = th.einsum('...ij, ...j -> ...i', R, v0[..., :2])

        q = th.sqrt(MAX_ACCEL * th.abs(x1) + v0**2 / 2)
        k1 = -(v0 - q) / MAX_ACCEL
        k2 = q / MAX_ACCEL
        wrench = th.sign(x1) * th.where(k1 > 0,
                                        +MAX_WRENCH,
                                        -MAX_WRENCH)
        wrench[..., :2] = th.einsum('...ji, ...i -> ...j', R,
                                    wrench[..., :2]) * 100

        # target velocity; proportional to residual
        # tgt_vel = th.zeros_like(s[..., 7:13])
        # tgt_vel[..., :2] = K_VEL * (goal_2d - obj_pos[..., :2])
        # vel_err = tgt_vel - obj_vel
        # wrench = K_WRENCH * vel_err
        return wrench


class WrenchGenerator:
    def __init__(self,
                 num_env: int, device: th.device,
                 policy: nn.Module):
        self.num_env = num_env
        self.device = device
        self.policy = policy.to(device)
        # self.policy.net = self.policy.net.to(device)

    def __call__(self, obs: Dict[str, th.Tensor]):
        # TODO:
        # reminder: also needs to concat the goal to the state
        action = self.policy.get_action(
            {'goal': obs['goal'],
             'state': obs['object_state']},
            sample=False,
            raw_obs=True,
            clip_action=True)
        # ic('wrenchgen', action)

        # We need to convert the ("normalized") action
        # into wrench.
        # FIXME: manually applying the
        # effects of `scale_action... despicable
        # wrench = action.detach().clone()
        # wrench[..., -6:-3] *= 10.0  # 1e+2
        # wrench[..., -3:] *= 0.1  # 1e-3
        return action


def _load_wrench_policy(num_env: int, device: str):
    path = Path('/tmp/pkm/ppo/run-012/')
    cfg_file = path / 'cfg.json'
    ppo_cfg = PPO.Config()

    with open(str(cfg_file), 'r') as fp:
        c = json.load(fp)
        ppo_cfg = recursive_replace_map(
            PPO.Config.from_dict(c['ppo']),
            {
                'cts_policy_type': 'continuous',
                'train.train_steps': 0,
                'device': device
            }
        )
    wrench_policy = PPO(ppo_cfg, env=None)
    wrench_policy.load(
        last_ckpt(path / 'ckpt')
    )
    wrench_policy = wrench_policy.to(device)

    wrench_policy.eval()
    # FIXME: either make PPO a torch module
    # or figure out a way to avoid introspection.
    # wrench_policy.net.eval()
    for p in wrench_policy.parameters():
        p.requires_grad_(False)
    return wrench_policy
