import pickle
from isaacgym import gymtorch
import numpy as np
from util.torch_util import dcn

class TrajRecorder():
    def __init__(self, num_envs):
        self.franka_link_dict = {'panda_hand': 8, 'panda_leftfinger': 9, 'panda_link0': 0, 
                                 'panda_link1': 1, 'panda_link2': 2, 'panda_link3': 3, 'panda_link4': 4, 'panda_link5': 5, 
                                 'panda_link6': 6, 'panda_link7': 7, 'panda_rightfinger': 10}
        self.box_idxs = [1]
        self.num_envs = num_envs
        self.num_steps = 500 * 8 # Specify the number of steps to run
        self.mix_reset_step_cnt = 0 * 8
        self.step_cnt = 0
        self.collected_data_positions = {name: [] for name in self.franka_link_dict.keys()}  # Store position data for each joint
        self.collected_data_orientations = {name: [] for name in self.franka_link_dict.keys()}  # Store orientation data for each joint
        self.collected_data_positions['object']=[]      # Add object
        self.collected_data_orientations['object']=[]
        self.collected_data_positions['target']=[]      # Add goal
        self.collected_data_orientations['target']=[]

        self.output_file = '/home/user/DyWA/output/record/traj/test.pkl'

        '''
        def __allocate_array(self, dat, is_position):
            for name in dat:
                if is_position:
                    dat[name] = np.zeros((self.num_steps,))        #NOTE: shape (num_frames, num_envs, 3) for position data
                else:
                    ...
        '''

    def record(self, gym, sim, goal = None):
        self.step_cnt += 1
        if self.step_cnt <= self.mix_reset_step_cnt:
            return

        if self.step_cnt <= self.num_steps + self.mix_reset_step_cnt:
            ...
        elif self.step_cnt == self.num_steps + self.mix_reset_step_cnt +1:
            self.export_cartesian_poses_pkl()
            return
        else:
            return

        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)
        gym.refresh_mass_matrix_tensors(sim)

        rb_states_new = gym.acquire_rigid_body_state_tensor(sim)
        rb_states_tensor = gymtorch.wrap_tensor(rb_states_new).to('cpu')        #NOTE: shape (num_envs*16,13)

        # Get joint positions and orientations for the step
        joint_positions, joint_orientations = self.get_cartesian_poses_for_joints_with_names(
            rb_states_tensor, self.franka_link_dict, self.box_idxs)

        # Step 4: Store the joint positions and orientations for this step
        for name in joint_positions:
            self.collected_data_positions[name].append(joint_positions[name])
            self.collected_data_orientations[name].append(joint_orientations[name])

        if goal is not None:
            self.record_goal(goal)
        
        # Continue with the rest of the self.simulation logic...

    def export_cartesian_poses_pkl(self):
        # Step 5: Convert collected data to NumPy arrays
        for name in self.collected_data_positions:
            self.collected_data_positions[name] = np.array(self.collected_data_positions[name])
            self.collected_data_orientations[name] = np.array(self.collected_data_orientations[name])

        # Step 6: Save the collected data to a .pkl file
        with open(self.output_file, "wb") as f:
            pickle.dump({
                "rigid_body_pos": self.collected_data_positions,
                "rigid_body_xyzw_quat": self.collected_data_orientations
            }, f)

        print(f"Data saved to {self.output_file}")

    def get_cartesian_poses_for_joints_with_names(self, rb_states, franka_link_dict, box_idxs, offset = 2):
        
        joint_positions = {name: [] for name in franka_link_dict.keys()}  # Initialize dictionary for positions
        joint_orientations = {name: [] for name in franka_link_dict.keys()}  # Initialize dictionary for orientations
        joint_positions['object']=[]      # Add object
        joint_orientations['object']=[]

        # Loop through all environments and extract joint data
        for i in range(self.num_envs):
            # Extract joint positions and orientations for the Franka robot
            for joint_name, joint_idx in franka_link_dict.items():
                offset = 2
                if joint_name != "box":  # Exclude box, as it will be handled separately later
                    joint_pos = rb_states[i*14 + joint_idx+offset, :3].cpu().numpy()  # Convert to NumPy
                    joint_rot = rb_states[i*14 + joint_idx+offset, 3:7].cpu().numpy()  # Convert to NumPy
                    joint_positions[joint_name].append(joint_pos)
                    joint_orientations[joint_name].append(joint_rot)
            
            # Extract the box's position and orientation
            box_pos = rb_states[i*14 + box_idxs[0], :3].cpu().numpy()  # Convert to NumPy
            box_rot = rb_states[i*14 + box_idxs[0], 3:7].cpu().numpy()  # Convert to NumPy
            joint_positions['object'].append(box_pos)
            joint_orientations['object'].append(box_rot)
        
        # Convert lists to NumPy arrays
        for name in joint_positions:
            joint_positions[name] = np.array(joint_positions[name])
            joint_orientations[name] = np.array(joint_orientations[name])

        
        return joint_positions, joint_orientations    
    
    def record_goal(self, goal):
        goal_pos = dcn(goal[:, :3])
        goal_object_xyzw_quat = dcn(goal[:, 3:7])

        self.collected_data_positions['target'].append(goal_pos)
        self.collected_data_orientations['target'].append(goal_object_xyzw_quat)
