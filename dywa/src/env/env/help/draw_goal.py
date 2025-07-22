#!/usr/bin/env python3

import torch as th


class DrawGoal:
    def __init__(self):
        pass

    def reset(self, env: 'NvdrRecordViewer'):
        self.env = env

    def __call__(self, img: th.Tensor):
        env = self.env
        goal3 = env.task.goal  # Nx3

        # to homogeneous
        goal4 = th.empty((*goal3.shape[:-1], 4),
                         dtype=img.dtype,
                         device=img.device)
        goal4[..., :3] = goal3
        goal4[..., 3] = 1

        # get ndc transform
        T_cam = env.img_env.cam.renderer.T_cam  # Nx4x4
        goal_pt = th.einsum('...ij,...j->...i', T_cam, goal4)
        goal_xy = goal_pt[..., :2]
        goal_z = goal_pt[..., 2:3]
        # NDC -> pixel coord
        goal_xy.div_(goal_z).add_(1.0).mul_(0.5)  # .mul_(img.shape[-2:])
        goal_xy[..., 0].mul_(img.shape[-2])
        goal_xy[..., 1].mul_(img.shape[-1])
        goal_xy = goal_xy.to(th.int64)
        # img[..., goal_xy[..., 0], goal_xy[..., 1]] = 1
        img[th.arange(img.shape[0], device=img.device),
            :, goal_xy[..., 1], goal_xy[..., 0]] = 1
