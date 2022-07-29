import torch
from torch.utils.data import DataLoader
import nfe.experiments.latent_ode.lib.utils as utils
import logging
import os
import numpy as np
from tqdm import tqdm
from dm_control import suite
import argparse

from dataset import HopperDataset
from env import UnimalEnv
from models.flow_model import FlowModel
from utils import *
from experiment import Experiment

class HopperExperiment(Experiment):
    def __init__(self, runs=1000, mode='s0'):
        self.patience = 10
        dataset_obj = HopperDataset(n_runs=runs)
        obs, actions = dataset_obj.get_dataset()
        model = FlowModel(obs.shape[-1], actions.shape[-1], 128, 2, 'TimeTanh', 2, flow_model='gru', action_emb_dim=-1)
        super().__init__(dataset_obj, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='results/test.gif')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model_mode', type=str, default='auto', choices=['auto', 's0', 'truth'])

    args = parser.parse_args()
    print(f'save={args.save}')

    experiment = HopperExperiment(5000, mode=args.model_mode)
    if (args.mode == 'train'):
        experiment.train()
    experiment.finish(save=args.save)