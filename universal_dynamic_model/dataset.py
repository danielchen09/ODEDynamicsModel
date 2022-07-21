from torch.utils.data import Dataset
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch
from dm_control import suite

from env import UnimalEnv
from utils import *


class SingleEnvDataset(Dataset):
    def __init__(self, 
                 xml_path='../unimals_100/train/xml/vt-1409-9-14-02-22-02-11.xml', 
                 n_runs=1, 
                 n_steps=100, 
                 save_dir='data/single_env', 
                 regenerate=False, 
                 use_action=True, 
                 env=None):
        self.env = env
        if env is None:
            self.env = UnimalEnv(xml_path, padding=False)
        self.n_steps = n_steps
        self.n_runs = n_runs
        self.save_dir = save_dir
        self.use_action = use_action
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_path_prefix = f'{save_dir}/{get_filename(xml_path).split(".")[0]}_{n_runs}_{n_steps}'
        self.data_path = f'{self.save_path_prefix}_data.npy'
        self.actions_path = f'{self.save_path_prefix}_actions.npy'

        if os.path.exists(self.data_path) and os.path.exists(self.actions_path) and not regenerate:
            self.data = np.load(self.data_path, allow_pickle=True)
            self.actions = np.load(self.actions_path, allow_pickle=True)
        else:
            data, action = self.generate_run()
            self.data = np.zeros((n_runs, *data.shape))
            self.actions = np.zeros((n_runs, *action.shape))
            for run in tqdm(range(n_runs), desc='generating runs'):
                data, action = self.generate_run()
                self.data[run] = data
                self.actions[run] = action
            np.save(self.data_path, self.data, allow_pickle=True)
            np.save(self.actions_path, self.actions, allow_pickle=True)

    def get_action(self, use_action=True):
        action = self.sample_action()
        if not self.use_action or not use_action:
            action = np.zeros_like(action)
        return action

    def sample_action(self):
        return self.env.action_space.sample()
        
    def get_obs(self):
        return self.env.get_obs().reshape(1, -1)

    def generate_run(self):
        actions = []
        xs = []

        self.env.reset()
        xs.append(self.get_obs())
        actions.append(self.get_action(False))
        for t in range(self.n_steps - 1):
            action = self.get_action()
            actions.append(action)
            self.env.step(action)
            xs.append(self.get_obs())
        return np.vstack(xs), np.vstack(actions)
    
    def get_dataset(self):
        return torch.from_numpy(self.data).float(), torch.from_numpy(self.actions).float()
    

class SingleEnvJointDataset(SingleEnvDataset):
    def __init__(self, 
                 xml_path='../unimals_100/train/xml/vt-1409-9-14-02-22-02-11.xml', 
                 n_runs=1, 
                 n_steps=100, 
                 save_dir='data/single_env', 
                 regenerate=False, 
                 use_action=True, 
                 env=None):
        super().__init__(
            xml_path=xml_path,
            n_runs=n_runs,
            n_steps=n_steps,
            save_dir=save_dir,
            regenerate=regenerate,
            use_action=use_action,
            env=env
        )
    
    def get_obs(self):
        return self.env.get_simple_obs()


class HopperDataset(SingleEnvJointDataset):
    def __init__(self, 
                 xml_path='hopper', 
                 n_runs=1, 
                 n_steps=100, 
                 save_dir='data/hopper', 
                 regenerate=False, 
                 use_action=True):
        env = suite.load('hopper', 'stand')
        self.random_state = np.random.RandomState(42)
        super().__init__(
            xml_path=xml_path,
            n_runs=n_runs,
            n_steps=n_steps,
            save_dir=save_dir,
            regenerate=regenerate,
            use_action=use_action,
            env=env
        )
    
    def get_obs(self):
        return np.hstack([
            self.env.physics.data.qpos,
            self.env.physics.data.qvel
        ])
    
    def sample_action(self):
        action_spec = self.env.action_spec()
        return self.random_state.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate dataset')
    parser.add_argument('--env', type=str, default='single', choices=['single', 'joint'])
    parser.add_argument('--action', type=str, default='T', choices=['T', 'F'])
    parser.add_argument('--xml', type=str, default='vt-1409-9-14-02-22-02-11.xml')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='data/single_env')
    parser.add_argument('--regen', type=str, default='T', choices=['T', 'F'])
    args = parser.parse_args()

    if args.env == 'single':
        env = SingleEnvDataset
    elif args.env == 'joint':
        env = SingleEnvJointDataset
    ds = env(
        f'../unimals_100/train/xml/{args.xml}', 
        args.runs, 
        args.steps, 
        regenerate=args.regen == 'T', 
        save_dir=args.save_dir, 
        use_action=args.action == 'T')
    breakpoint()