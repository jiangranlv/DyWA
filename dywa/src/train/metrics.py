#!/usr/bin/env python3

from util.math_util import quat_from_axa
import torch as th
nn = th.nn
F = nn.functional


def sigmoid_inverse(x: th.Tensor) -> th.Tensor:
    """ Inverse of the sigmoid function. """
    x = th.as_tensor(x)
    return th.log(x / (1 - x))


class IoU(nn.Module):
    """
    Pointwise IoU (intersection over union) metric.
    """

    def __init__(self,
                 min_logit: float = 0.0,
                 eps: float = 1e-6):
        """
        Args:
            min_logit: Minimum logit value to be considered occupied.
                Usually computed via inverse_sigmoid(x).
        """
        super().__init__()
        self.min_logit = min_logit
        self.eps = eps

    def forward(self, input: th.Tensor, target: th.Tensor) -> th.Tensor:
        """
        Args:
            input: prediction (logits, >=`min_logit` is considered positive.)
            target: target (binary or float, >=0.5 is considered positive.)

        Returns:
            iou: The computed IoU metric.
        """
        x = (input >= self.min_logit)  # bool
        y = (target >= (0.5 + self.eps))  # bool
        ixn = th.logical_and(x, y)
        uxn = th.logical_or(x, y)
        iou = th.div(th.count_nonzero(ixn), th.count_nonzero(uxn))
        return iou


def pose_error(pose1, pose2):
    """
    计算两组姿态的误差，包括位置误差和旋转误差。
    
    参数:
    - pose1: 形状为 [n, 6] 的张量, 包含 [position, axis-angle] 
    - pose2: 形状为 [n, 6] 的张量, 包含 [position, axis-angle]
    
    返回:
    - pos_error: 形状为 [n] 的张量, 表示位置误差
    - rot_error: 形状为 [n] 的张量, 表示旋转误差（弧度）
    """
    # 拆分位置和轴角
    pos1, axis_angle1 = pose1[:, :3], pose1[:, 3:]
    pos2, axis_angle2 = pose2[:, :3], pose2[:, 3:]
    
    # 位置误差 (欧氏距离)
    pos_error = th.norm(pos1 - pos2, dim=1)
    
    # 旋转误差: 轴角转换为旋转矩阵
    
    q1 = quat_from_axa(axis_angle1)
    q2 = quat_from_axa(axis_angle2)
    
    # 计算旋转误差 (四元数之间的夹角)
    dot_product = th.sum(q1 * q2, dim=1)
    dot_product = th.clamp(dot_product, -1.0, 1.0)  # 限制值域在[-1, 1]
    rot_error = th.rad2deg(2 * th.acos(th.abs(dot_product)))
    
    return pos_error, rot_error
