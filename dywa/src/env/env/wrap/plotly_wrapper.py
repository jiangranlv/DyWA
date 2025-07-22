import os
from matplotlib import pyplot as plt
import numpy as np
import gym
from pathlib import Path
from dataclasses import dataclass
from env.env.wrap.normalize_env import NormalizeEnv
from util.config import ConfigBase
from env.env.iface import EnvIface
from env.env.wrap.base import WrapperEnv, ObservationWrapper
from util.math_util import pose7d_to_matrix, pose9d_to_matrix, rot6d_to_matrix
from util.path import ensure_directory
# from hacman.utils.transformations import transform_point_cloud, decompose_pose_mat
from util.torch_util import dcn
import wandb
import torch as th
import plotly.graph_objects as go

def format_color(color):
    if type(color) == str:
        color_str = color
    elif type(color) == np.ndarray and len(color) == 3:
        color = color.astype(np.int32)
        color_str = f'rgb({color[0]},{color[1]},{color[2]})'
    else:
        color = color.astype(np.int32)
        color_str = [f'rgb({color[k, 0]},{color[k, 1]},{color[k, 2]})' for k in range(color.shape[0])]
    return color_str

def plot_pcd(name, points, color, size=3):
    return go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", 
                        marker = dict(color=format_color(color), size=size), name=name)

def plot_pcd_with_score(name, points, action_score, size=3):
    assert type(action_score) == np.ndarray
    action_score = action_score.reshape(-1)
    action_score = (action_score - action_score.min())/(action_score.max()-action_score.min()+1e-7)
    object_color_id = (action_score*255).astype(np.int32)
    object_color = []
    for cid in object_color_id:
        try:
            object_color.append(plt.get_cmap('plasma').colors[cid])
        except:
            print(f'cid={cid} gives an error. Will use 0 instead.')
            object_color.append(plt.get_cmap('plasma').colors[0])
    object_color = np.array(object_color)*255
    return plot_pcd(name, points, object_color, size)

def plot_action(name, start, direction, color='red', size=3):
    direction = direction*0.02*3  # action scale=0.02. steps=10
    if start is None: x, y, z = 0, 0, 0
    else: x, y, z = start[0], start[1], start[2]
    u, v, w = direction[0], direction[1], direction[2]
    return go.Scatter3d(x=[x, x + u], y=[y, y + v], z=[z, z + w], mode='lines',
                        line=dict(color=format_color(color), width=10), name=name)


def plot_pose(name, pose, axis_length = 0.1, width = 6):
    position = pose[:3, 3]
    rotation_matrix = pose[:3, :3]
    axes = [
        # X 轴 - 红色
        go.Scatter3d(
            x=[position[0], position[0] + rotation_matrix[0,0]*axis_length],
            y=[position[1], position[1] + rotation_matrix[1,0]*axis_length],
            z=[position[2], position[2] + rotation_matrix[2,0]*axis_length],
            mode='lines',
            line=dict(color='red', width=6),
            name=f'{name} X axis'
        ),
        # Y 轴 - 绿色
        go.Scatter3d(
            x=[position[0], position[0] + rotation_matrix[0,1]*axis_length],
            y=[position[1], position[1] + rotation_matrix[1,1]*axis_length],
            z=[position[2], position[2] + rotation_matrix[2,1]*axis_length],
            mode='lines',
            line=dict(color='green', width= width),
            name=f'{name} Y axis'
        ),
        # Z 轴 - 蓝色
        go.Scatter3d(
            x=[position[0], position[0] + rotation_matrix[0,2]*axis_length],
            y=[position[1], position[1] + rotation_matrix[1,2]*axis_length],
            z=[position[2], position[2] + rotation_matrix[2,2]*axis_length],
            mode='lines',
            line=dict(color='blue', width= width),
            name=f'{name} Z axis'
        )
    ]
    return axes

class PlotlyPointCloudRecorder(WrapperEnv):
    """
    修改后的 WandbPointCloudRecorder 类，仅保留 Plotly 可视化及点云功能。
    """
    
    @dataclass
    class Config(ConfigBase):
        save_plotly: bool = True
        record_dir: str = '/tmp/pkm/plotly'
        real_robot: bool = False
        log_plotly_once: bool = True

    def __init__(self, cfg: Config, env: EnvIface):
        super().__init__(env)
        self.config = cfg
        self.real_robot = cfg.real_robot
        self.log_plotly_once = cfg.log_plotly_once
        
        self._record_dir: Path = ensure_directory(
            cfg.record_dir)
        
        self.vis = []
        self.recording = True
        self.title = None
        self.step_count = 0
        self.plot_count = 0
        self.episode_count = -1

        self.maximum = 100

        norm_env = self.unwrap(target=NormalizeEnv)
        self.normalizer = norm_env.normalizer
        self.target = ['partial_cloud','abs_goal', 'trans_cloud',
                       'object_state', 'goal_cloud']

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space
    
    def reset(self):
        obs = self.env.reset()
        # self.reset_plotly_logging(obs)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.step_count += 1
        if self.recording and self.step_count< 400 and self.step_count >= 0: #### TODO
        # if self.recording and self.step_count < self.maximum:
            for vis_index in range(self.env.num_env):
                self.plot_obs(obs, vis_index)

        self.prev_obs = obs
        return obs, rew, done, info

    def plot_obs(self, obs, vis_index):
        self.vis = []
        unnorm_obs = {key: obs[key] for key in obs if key in self.target}
        unnorm_obs = self.normalizer.unnormalize_obs(unnorm_obs)

        object_pose = pose7d_to_matrix(unnorm_obs['object_state'][:, :7])
        goal_pose = pose9d_to_matrix(unnorm_obs['abs_goal'])
        self.vis.extend(plot_pose(f'pose_{self.step_count}',dcn(object_pose[vis_index])))
        self.vis.extend(plot_pose(f'goal_{self.step_count}',dcn(goal_pose[vis_index])))

        colors = ['deeppink', 'lightblue', 'darkgreen', 'red', 'orange', 'purple', 'yellow']  
        for i, key in enumerate(self.target):
            color = colors[i % len(colors)] 
            if key == 'trans_cloud':
                rel_goal = th.bmm(goal_pose, th.linalg.inv(object_pose))
                goal_pc = th.bmm(unnorm_obs['partial_cloud'] ,rel_goal[:, :3, :3].transpose(1, 2)) + rel_goal[:, :3, 3].unsqueeze(1)
                self.vis.append(plot_pcd(f'trans_pc_{self.step_count}', dcn(goal_pc[vis_index]), color)) 

            elif 'cloud' in key:
                self.vis.append(plot_pcd(f'{key}_{self.step_count}', dcn(unnorm_obs[key][vis_index]), color))  

        if self.config.save_plotly:
            filename = f"plotly_step{self.step_count}_env{vis_index}.html"
            fig = go.Figure(self.vis)
            fig.update_layout(
                    title="3D pc",
                    scene=dict(
                        xaxis=dict(title='X axis'),
                        yaxis=dict(title='Y axis'),
                        zaxis=dict(title='Z axis'),
                        aspectmode='data'  # 使所有轴的比例与数据一致，保持等距
                    )
                )
            fig.write_html(os.path.join(self._record_dir, filename), auto_play=False)  # 本地保存为 HTML


    def step_old(self, action):
        obs, rew, done, info = self.env.step(action)
        
        if self.recording:
            self.title = f"Final reward: {rew:.2f} success: {info.get('is_success', False)}"
            self.step_count += 1
            action_location = info.get('action_location', [None])[0]

            if action_location is None:
                # 使用前一个观测中的抓手位置
                if not hasattr(self, "prev_obs") or self.prev_obs is None:
                    action_location = np.zeros(3)
                else:
                    prev_gripper_pose = self.prev_obs.get('gripper_pose', [np.zeros(12)])[0]
                    action_location, _ = decompose_pose_mat(prev_gripper_pose)
            
            if 'poke_success' in info and not info['poke_success']:
                self.vis.append(plot_action(f'a_{self.step_count-1}', action_location, info.get('action_param', []), color='orange'))
            else:
                self.vis.append(plot_action(f'a_{self.step_count-1}', action_location, info.get('action_param', [])))                
                    
            if done:
                obj_pcd = info.get('terminal_observation', {}).get('object_pcd_points', [])[0]
                if obj_pcd is not None:
                    self.vis.append(plot_pcd(f'o_final', obj_pcd, 'mediumpurple'))
                self.close_plotly_recorder()
                if not self.log_plotly_once:
                    self.reset_plotly_logging(obs)                
            else:
                if self.real_robot:
                    # 估计的目标点云
                    goal_pcd = transform_point_cloud(obs['object_pose'][0], obs['goal_pose'][0], obs['object_pcd_points'][0])
                    self.vis.append(plot_pcd(f'g_{self.step_count}', goal_pcd, 'lightblue'))
                else:
                    goal_pcd = transform_point_cloud(obs['object_pose'][0], obs['goal_pose'][0], obs['object_pcd_points'][0])
                    self.vis.append(plot_pcd(f'g_{self.step_count}', goal_pcd, 'blue'))
                        
                if 'action_location_score' in obs:
                    self.vis.append(plot_pcd_with_score(f'o_{self.step_count}', obs['object_pcd_points'][0], obs['action_location_score'][0]))
                else:
                    self.vis.append(plot_pcd(f'o_{self.step_count}', obs['object_pcd_points'][0], 'purple'))
                self.vis.append(plot_pcd(f'o_next_{self.step_count-1}', obs['object_pcd_points'][0], 'yellow'))
        
        self.prev_obs = obs
        return obs, rew, done, info

    def reset_plotly_logging(self, obs):
        self.recording = True
        self.vis = []
        self.title = None        
        self.step_count = 0
        self.episode_count += 1
        self.vis.append(plot_pcd('background', obs.get('background_pcd_points', [])[0], 'lightgrey', size=2))
    
        if self.real_robot:
            # 地面真实目标点云
            goal_pcd = self.env.unwrapped.env.goal_pcd.voxel_down_sample(0.01)
            goal_pcd = np.asarray(goal_pcd.points)
            self.vis.append(plot_pcd(f'goal', goal_pcd, 'blue'))
            # 估计的目标点云
            goal_pcd = transform_point_cloud(obs['object_pose'][0], obs['goal_pose'][0], obs['object_pcd_points'][0])
            self.vis.append(plot_pcd(f'g_0', goal_pcd, 'lightblue'))
        else:
            goal_pcd = transform_point_cloud(obs['object_pose'][0], obs['goal_pose'][0], obs['object_pcd_points'][0])
            self.vis.append(plot_pcd(f'g_0', goal_pcd, 'blue'))
        
        if 'action_location_score' in obs:
            self.vis.append(plot_pcd_with_score(f'o_0', obs['object_pcd_points'][0], obs['action_location_score'][0]))
        else:
            self.vis.append(plot_pcd(f'o_0', obs['object_pcd_points'][0], 'yellow'))

    def close_plotly_recorder(self):
        if self.recording:
            global_step = self.episode_count
            fig = go.Figure(self.vis)
            # fig = self.get_plotly_with_slidebar()
            # wandb.log({"visualizations": fig, 'global_steps': global_step})

            if self.config.save_plotly:
                filename = f"plotly_{global_step}_{self.plot_count}.html"
                fig.write_html(os.path.join(self._record_dir, filename), auto_play=False)  # 本地保存为 HTML
                self.plot_count += 1

            self.recording = False
            self.vis = []
            self.title = None
            self.step_count = 0

    def get_plotly_with_slidebar(self):
        vis_name2id = {vis.name: idx for idx, vis in enumerate(self.vis)}
        fig = go.Figure(self.vis)
        
        # 默认隐藏所有数据
        for trace in fig.data:
            trace.visible = False
        # 显示基础元素
        fig.data[vis_name2id['background']].visible = True    
        if 'goal' in vis_name2id:
            fig.data[vis_name2id['goal']].visible = True 
        fig.data[vis_name2id['g_0']].visible = True
        fig.data[vis_name2id['o_0']].visible = True
        fig.data[vis_name2id['a_0']].visible = True

        fig.update_scenes(aspectmode='data')
        fig.update_layout(title=self.title)

        steps = []
        for i in range(self.step_count):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title": self.title}],
            )
            step["args"][0]["visible"][vis_name2id['background']] = True
            if 'goal' in vis_name2id:
                step["args"][0]["visible"][vis_name2id['goal']] = True
            if f'g_{i}' in vis_name2id:
                step["args"][0]["visible"][vis_name2id[f'g_{i}']] = True                
            step["args"][0]["visible"][vis_name2id[f'o_{i}']] = True
            step["args"][0]["visible"][vis_name2id[f'a_{i}']] = True

            if i == self.step_count - 1:
                if 'o_final' in vis_name2id:
                    step["args"][0]["visible"][vis_name2id['o_final']] = True
            else:
                step["args"][0]["visible"][vis_name2id.get(f'o_next_{i}', False)] = True
            steps.append(step)

        sliders = [dict(
            active=0,    
            pad={"t": 50},
            steps=steps
        )]
        fig.update_layout(sliders=sliders)
        return fig

    def close(self):
        self.close_plotly_recorder()
