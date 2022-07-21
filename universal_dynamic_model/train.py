from nfe.experiments.base_experiment import BaseExperiment
from nfe.experiments.latent_ode.lib.create_latent_ode_model import create_LatentODE_model
from nfe.experiments.latent_ode.lib.parse_datasets import parse_datasets
from nfe.experiments.latent_ode.lib.utils import compute_loss_all_batches
import nfe.experiments.latent_ode.lib.utils as utils
import torch
import numpy as np
from torch.utils.data import DataLoader
import logging

from dataset import HopperDataset, SingleEnvJointDataset
from parse_arguments import parse_arguments
from env import UnimalEnv
from utils import *

class LatentODEExperiment(BaseExperiment):
    def get_model(self, args): 
        z0_prior = torch.distributions.Normal(
            torch.Tensor([0.0]).to(self.device),
            torch.Tensor([1.]).to(self.device)
        )

        obsrv_std = 0.001 if args.data == 'hopper' else 0.01
        obsrv_std = torch.Tensor([obsrv_std]).to(self.device)

        model = create_LatentODE_model(args, self.dim, z0_prior, obsrv_std, self.device, n_labels=self.n_classes)
        return model

    def get_data(self, args):
        device = args.device
        
        if args.data == 'hopper':
            dataset_obj = HopperDataset(n_runs=1000)
        else:
            dataset_obj = SingleEnvJointDataset(n_runs=1000, xml_path=args.data)

        obs, actions = dataset_obj.get_dataset()[:args.n] # obs, action [b, t, d]

        actions = torch.cat([actions.clone()[:, 1:, :], torch.zeros(actions.shape[0], 1, actions.shape[-1])], dim=1)
        dataset = torch.cat([actions, obs.clone()], dim=-1)
        action_dim = actions.shape[-1]
        dataset = dataset.to(device)

        def basic_collate_fn(batch, time_steps, args=args, device=device, data_type='train'):
            batch = torch.stack(batch)
            data_dict = {
                'data': batch[:, :-1, :],
                'time_steps': time_steps.unsqueeze(0)
            }

            data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
            data_dict['data_to_predict'] = batch[:, 1:, action_dim:]
            return data_dict


        n_tp_data = dataset[:].shape[1] - 1

        # Time steps that are used later on for exrapolation
        time_steps = torch.arange(start=0, end=n_tp_data, step=1).float().to(device)
        time_steps = time_steps / len(time_steps)

        time_steps = time_steps.to(device)

        if not args.extrap:
            # Creating dataset for interpolation
            # sample time points from different parts of the timeline,
            # so that the model learns from different parts of hopper trajectory
            n_traj = len(dataset)
            n_tp_data = dataset.shape[1]
            n_reduced_tp = args.timepoints

            # sample time points from different parts of the timeline,
            # so that the model learns from different parts of hopper trajectory
            start_ind = np.random.randint(0, high=n_tp_data - n_reduced_tp +1, size=n_traj)
            end_ind = start_ind + n_reduced_tp
            sliced = []
            for i in range(n_traj):
                  sliced.append(dataset[i, start_ind[i] : end_ind[i], :])
            dataset = torch.stack(sliced).to(device)
            time_steps = time_steps[:n_reduced_tp]

        train_y, val_y, test_y = utils.split_train_val_test(dataset)

        n_samples = len(dataset)
        input_dim = actions.shape[-1] + obs.shape[-1]
        output_dim = obs.shape[-1]

        dltrain = DataLoader(train_y, batch_size=args.batch_size, shuffle=True,
            collate_fn=lambda batch: basic_collate_fn(batch, time_steps, data_type='train'))
        dlval = DataLoader(val_y, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda batch: basic_collate_fn(batch, time_steps, data_type='test'))
        dltest = DataLoader(test_y, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda batch: basic_collate_fn(batch, time_steps, data_type='test'))
        return input_dim, output_dim, dltrain, dlval, dltest

    def training_step(self, batch):
        loss = self.model.compute_all_losses(batch)
        return loss['loss']

    def _get_loss(self, dl):
        loss = compute_loss_all_batches(self.model, dl, self.args, self.device)
        return loss['loss'], loss['mse'], loss['acc']

    def validation_step(self):
        loss, mse, acc = self._get_loss(self.dlval)
        self.logger.info(f'val_mse={mse:.5f}')
        self.logger.info(f'val_acc={acc:.5f}')
        return loss

    def test_step(self):
        loss, mse, acc = self._get_loss(self.dltest)
        self.logger.info(f'test_mse={mse:.5f}')
        self.logger.info(f'test_acc={acc:.5f}')
        return loss

    def finish(self):
        import os
        import numpy as np
        from tqdm import tqdm
        from dm_control import suite

        if self.args.data == 'hopper':
            env = suite.load('hopper', 'stand')
        else:
            env = UnimalEnv(self.args.data)

        physics = env.physics

        model_path = 'saved_models/best_model.pt'
        if not os.path.exists(model_path):
            print('model path does not exist')
            return

        self.model.load_state_dict(torch.load(model_path))
        batch_dict = next(iter(self.dltest))
        
        pred_y, info = self.model.get_reconstruction(
            batch_dict['tp_to_predict'], 
            batch_dict['observed_data'], 
            batch_dict['observed_tp'], 
            mask=batch_dict['observed_mask'], 
            n_traj_samples=1, 
            mode=batch_dict['mode']
        )

        pred_y = pred_y.detach().cpu().numpy()[0][0]
        actual_y = batch_dict['data_to_predict'].detach().cpu().numpy()[0]
        print(pred_y.shape, actual_y.shape)

        frames_actual = []
        env.reset()
        with physics.reset_context():
            for t in tqdm(range(actual_y.shape[0]), 'generating actual'):
                physics.set_state(actual_y[t, :])
                physics.step()
                frames_actual.append(physics.render(camera_id=0))
        
        frames_pred = []
        env.reset()
        with physics.reset_context():
            for t in tqdm(range(pred_y.shape[0]), 'generating pred'):
                physics.set_state(pred_y[t, :])
                physics.step()
                frames_pred.append(physics.render(camera_id=0))
        
        frames = []
        for frame_actual, frame_pred in tqdm(zip(frames_actual, frames_pred), 'generating vid'):
            frames.append(np.hstack([frame_actual, frame_pred]))
        generate_video(frames)
        print('video saved')


if __name__ == '__main__':
    args = parse_arguments()

    logger = logging.getLogger()

    experiment = LatentODEExperiment(args, logger)

    # experiment.finish()
    experiment.train()
    experiment.finish()

    