import glob
import gym
from gym.spaces import Box, Dict
from dm_control.mujoco import Physics

from utils import *
from config import config

class UnimalEnvSampler:
    def __init__(self, xml_dir, num_envs=-1):
        # num_envs = -1 if use all
        self.xml_paths = glob.glob(f'{xml_dir}/*.xml')
        if num_envs != -1:
            self.xml_paths = self.xml_paths[:num_envs]
        self.init = False

    def _init_error(self, errors):
        self.init = True
        self.errors = {xml_path: errors[0] for xml_path in self.xml_paths}

    def get_probs(self):
        pass

    def sample_env(self):
        pass

    def update_error(self, envs, errors):
        if not self.init:
            self._init_error(errors)
        for env, error in zip(envs, errors):
            self.errors[env] = config.buffer


class UnimalEnv(gym.Env):
    def __init__(self, xml_path, padding=True):
        super().__init__()
        self.xml_path = xml_path
        self.padding = padding

        self.physics = Physics.from_xml_path(xml_path)
        self.env_metadata = get_unimal_metadata(xml_path)
        self.xml_root = parse_xml(xml_path).getroot()

        self.traversal_order = tree_traversal(self.xml_root)
        self.body_order = [name2id(self.physics, 'body', get_property(x, 'name')) for x in self.traversal_order]
        self.body_order = np.array(self.body_order) - 1

        self.body_idxs = get_idxs(self.physics, 'body')
        self.geom_idxs = get_idxs(self.physics, 'geom')

        self.body_names = get_names(self.physics, 'body')[1:]
        self.n_body = len(self.body_names)
        self.joint_names = get_names(self.physics, 'jnt', lambda x: 'limb' in x)

        # obs space = n_limbs x (limb_obs + 2 * joint_obs)
        sample_obs = self.get_obs()
        if self.padding:
            sample_obs, sample_mask = sample_obs
        self.observation_shape = sample_obs.shape
        self.observation_space = Dict({
            'obs': Box(
                low=np.full(sample_obs.shape, -float("inf"), dtype=np.float32),
                high=np.full(sample_obs.shape, float("inf"), dtype=np.float32),
                dtype=sample_obs.dtype
            )
        })

        action_low, action_high = get_action_bound(self.physics)
        self.action_space = Box(
            low=action_low,
            high=action_high,
            dtype=np.float32
        )
    
    def step(self, action):
        self.physics.set_control(action)
        self.physics.step()
        return self.get_obs(), 0, False # obs, reward, done

    def reset(self):
        self.physics.reset()
        return self.get_obs()

    def render(self):
        return self.physics.render(camera_id=0)
    
    def get_simple_obs(self):
        return np.hstack([
            self.physics.data.qpos.copy(),
            self.physics.data.qvel.copy()
        ])

    def get_obs(self):
        def select_obs(obs):
            obs_arr = []
            for obs_name in config.env.SELECT_OBS:
                if obs_name in obs:
                    obs_arr.append(obs[obs_name])
            return np.hstack(obs_arr)

        body_obs = select_obs(self.get_body_obs())
        joint_obs = select_obs(self.get_joint_obs())
        joint_obs_size = joint_obs.shape[-1]

        joint_mask = self.get_joint_mask()
        obs = np.zeros((self.n_body, joint_obs_size * 2))
        obs = obs.reshape(-1, joint_obs_size)
        obs[joint_mask] = joint_obs
        obs = obs.reshape(self.n_body, joint_obs_size * 2)
        obs = np.hstack((body_obs, obs))
        obs = obs[self.body_order]
        if not self.padding:
            return obs
        padding = np.zeros((config.model.MAX_SEQ_LEN - obs.shape[0], obs.shape[-1]))
        mask = np.array([False] * obs.shape[0] + [True] * padding.shape[0])
        obs = np.vstack([obs, padding])
        return obs, mask

    
    def get_joint_mask(self):
        mask_dict = {body_name: [False, False] for body_name in self.body_names}
        for joint_name in self.joint_names:
            joint_type, joint_idx = joint_name.split('/')
            joint_type = joint_type[-1]
            mask_dict[f'limb/{joint_idx}'][['x', 'y'].index(joint_type)] = True
        return np.array([mask_dict[body_name] for body_name in self.body_names]).flatten()

    def get_body_obs(self):
        obs = {}

        # geom dynamic data
        obs['geom_xpos'] = self.physics.data.geom_xpos[self.geom_idxs, :].copy()
        obs['geom_xvel'] = get_vel(self.physics, self.geom_idxs, 'geom')
        obs['geom_xquat'] = mat2quat(self.physics.data.geom_xmat[self.geom_idxs, :])

        # body dynamic data
        obs['body_xpos'] = self.physics.data.xpos[self.body_idxs, :].copy()
        obs['body_xquat'] = self.physics.data.xquat[self.body_idxs, :].copy()
        obs['body_xvel'] = get_vel(self.physics, self.body_idxs, 'body')

        # geom static data
        obs['geom_type'] = one_hot(self.physics.model.geom_type[self.geom_idxs], 8)
        obs['geom_pos'] = self.physics.model.geom_pos[self.geom_idxs, :].copy()
        obs['geom_quat'] = self.physics.model.geom_quat[self.geom_idxs, :].copy()
        obs['geom_size'] = self.physics.model.geom_size[self.geom_idxs, :].copy()
        obs['geom_friction'] = self.physics.model.geom_friction[self.geom_idxs, :].copy()

        # body static data
        obs['body_pos'] = self.physics.model.body_pos[self.body_idxs, :].copy()
        obs['body_ipos'] = self.physics.model.body_ipos[self.body_idxs, :].copy()
        obs['body_iquat'] = self.physics.model.body_iquat[self.body_idxs, :].copy()
        obs['body_mass'] = self.physics.model.body_mass[self.body_idxs, None].copy()

        return obs

    def get_joint_obs(self):
        obs = {}

        # joint dynamic data
        qpos = self.physics.data.qpos[7:, None].copy()
        jnt_range = self.physics.model.jnt_range[1:].copy()
        qpos = (qpos - jnt_range[:, 0:1]) / (jnt_range[:, 1:2] - jnt_range[:, 0:1])
        obs['jnt_qpos'] = qpos
        obs['jnt_qvel'] = self.physics.data.qvel[6:, None].copy()

        # joint static data
        obs['jnt_pos'] = self.physics.model.jnt_pos[1:].copy()
        obs['jnt_range'] = jnt_range
        obs['jnt_axis'] = self.physics.model.jnt_axis[1:].copy()
        obs['jnt_gear'] = self.physics.model.actuator_gear[:, 0:1].copy()
        obs['jnt_armature'] = self.physics.model.dof_armature[6:, None].copy()
        obs['jnt_damping'] = self.physics.model.dof_damping[6:, None].copy()

        return obs

    def get_global_obs(self):
        opt = self.physics.model.opt
        return np.array([[
            opt.timestep,
            *opt.gravity,
            *opt.wind,
            *opt.magnetic,
            opt.density,
            opt.viscosity,
            opt.impratio,
            opt.o_margin,
            *opt.o_solref,
            *opt.o_solimp
        ]])


if __name__ == '__main__':
    env_6_name = '../unimals_100/train/xml/vt-1409-9-14-02-22-02-11.xml'
    env_10_name = '../unimals_100/train/xml/mvt-5506-15-3-17-08-19-25.xml'
    env = UnimalEnv(env_6_name, padding=False)
    state = env.get_simple_obs()
    env.physics.set_state(state)