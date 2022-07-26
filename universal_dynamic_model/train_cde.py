import torch
from torch.utils.data import DataLoader
import nfe.experiments.latent_ode.lib.utils as utils
import logging
import os
import numpy as np
from tqdm import tqdm
from dm_control import suite

from dataset import HopperDataset
from env import UnimalEnv
from models.cde_model import CDEModel
from utils import *
from experiment import Experiment

class CDEExperiment(Experiment):
    def __init__(self):
        self.patience = 10
        dataset_obj = HopperDataset(n_runs=1000)
        obs, actions = dataset_obj.get_dataset()
        model = CDEModel(obs.shape[-1], actions.shape[-1], 128)
        super().__init__(dataset_obj, model)

if __name__ == '__main__':
    experiment = CDEExperiment()
    experiment.train()
    experiment.finish(save='results/cde.gif')