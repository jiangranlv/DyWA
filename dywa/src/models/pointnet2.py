# some code borrowed from https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import ball_query

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from util.config import ConfigBase

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    dists, idx, nn = ball_query(p1=new_xyz, p2=xyz, K=nsample,radius = radius)
    return idx

# def query_ball_point(radius, nsample, xyz, new_xyz):
#     """
#     Input:
#         radius: local region radius
#         nsample: max sample number in local region
#         xyz: all points, [B, N, 3]
#         new_xyz: query points, [B, S, 3]
#     Return:
#         group_idx: grouped points index, [B, S, nsample]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     _, S, _ = new_xyz.shape
#     group_idx = torch.arange(N, dtype=torch.long).to(
#         device).view(1, 1, N).repeat([B, S, 1])
#     sqrdists = square_distance(new_xyz, xyz)
#     group_idx[sqrdists > radius ** 2] = N
#     group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
#     group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
#     mask = group_idx == N
#     group_idx[mask] = group_first[mask]
#     return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        # [B, npoint, nsample, C+D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat(
                    [grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            # print(i, radius, new_points.shape)
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        # print(new_points_concat.shape)
        return new_xyz, new_points_concat

'''
self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], self.info_dim, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
hidden_dim = [512, 256]
'''

# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_cls_msg.py
class PointNet2Module(nn.Module):
    @dataclass
    class Config(ConfigBase):
        set_abstraction1_cfg: Optional[Dict[str, Any]] = None
        set_abstraction2_cfg: Optional[Dict[str, Any]] = None
        set_abstraction3_cfg: Optional[Dict[str, Any]] = None
        hidden_dim: Optional[Optional[List[int]]] = None

    def __init__(self, cfg: Config, output_dim: int, norm: str):
        super().__init__()
        self.cfg = cfg

        if norm == 'bn':
            norm_type = nn.BatchNorm1d
        elif norm == 'ln':
            norm_type = nn.LayerNorm
        else:
            raise NotImplementedError
        
        self.info_dim = self.cfg.set_abstraction1_cfg["in_channel"]
        self.output_dim = output_dim
        self.hidden_dim = self.cfg.hidden_dim

        self.sa1 = PointNetSetAbstractionMsg(**self.cfg.set_abstraction1_cfg)
        self.sa2 = PointNetSetAbstractionMsg(**self.cfg.set_abstraction2_cfg)
        self.sa3 = PointNetSetAbstraction(**self.cfg.set_abstraction3_cfg)

        if self.hidden_dim is not None:
            self.global_feature_dim = self.cfg.set_abstraction3_cfg["mlp"][-1]
            self.layer_n = len(self.hidden_dim) + 1
            self.mlp_dim = [self.global_feature_dim] + [i for i in self.hidden_dim] + [self.output_dim]
            self.fcs = nn.ModuleList([nn.Linear(self.mlp_dim[i], self.mlp_dim[i+1]) for i in range(self.layer_n)])
            self.bns = nn.ModuleList([norm_type(self.mlp_dim[i+1]) for i in range(self.layer_n-1)]) 

    def forward(self, xyz: torch.Tensor):
        """xyz: [B, N, 3 or d]"""
        B, N, D = xyz.shape
        xyz = xyz.view(B, D, N)
        if self.info_dim > 0:
            info = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            info = None
        l1_xyz, l1_points = self.sa1(xyz, info)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points
        if self.hidden_dim is not None:
            x = x.view(-1, self.global_feature_dim)
            for i in range(self.layer_n-1):
                x = F.relu(self.bns[i](self.fcs[i](x)))
            x = self.fcs[-1](x)
            # x = x.view(B, -1, self.output_dim)
        return x


# change the function to torch version with batch dimension and return mean and variance
def pc_normalize_torch(pc):
    # Compute the centroid for each batch
    centroid = torch.mean(pc, dim=1, keepdim=True)  # Shape: (batch_size, 1, 3)
    pc = pc - centroid  # Broadcasting works here
    # Normalize by the max distance from the origin
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=2)), dim=1)[0]  # Shape: (batch_size,)
    pc = pc / m.unsqueeze(1).unsqueeze(2)  # Broadcasting the max value to the shape (batch_size, num_points, 3)
    return pc, centroid.squeeze(1), m.unsqueeze(1)

if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('dywa/src/data/cfg/student/encoder/pointnet2.yaml')

    pointnet_config = PointNet2Module.Config(
        set_abstraction1_cfg=cfg.set_abstraction1_cfg,
        set_abstraction2_cfg=cfg.set_abstraction2_cfg,
        set_abstraction3_cfg=cfg.set_abstraction3_cfg,
        hidden_dim=cfg.hidden_dim
    )
    model = PointNet2Module(cfg.output_dim, pointnet_config)
    print(model)
    x = torch.randn(2, 1024, 3)
    pc_norm, centroid, scale = pc_normalize_torch(x)
    print(centroid.shape, scale.shape)
    y = model(x)
    print(y.shape)

        