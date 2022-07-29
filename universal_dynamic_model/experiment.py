import logging
import torch
from torch.utils.data import DataLoader
from dm_control import suite
import os
from tqdm import tqdm
import nfe.experiments.latent_ode.lib.utils as utils

from utils import *
from plotter import Plotter


class Experiment:
    def __init__(self, dataset_obj, model,
                 patience=10,
                 lr=1e-3,
                 weight_decay=0.0001,
                 lr_decay_step=20,
                 lr_dacay=0.5):
        self.patience = patience

        self.plotter = Plotter()
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        obs, actions = dataset_obj.get_dataset()

        # normalize obs
        self.jnt_range = dataset_obj.env.physics.model.jnt_range[3:]
        self.qpos_dim = dataset_obj.env.physics.data.qpos.shape[0]
        obs[:, :, 3:self.qpos_dim] -= self.jnt_range[:, 0]
        obs[:, :, 3:self.qpos_dim] /= self.jnt_range[:, 1] - self.jnt_range[:, 0]
        # breakpoint()

        action_dim = actions.shape[-1]
        actions = torch.cat([actions.clone()[:, 1:, :], torch.zeros(actions.shape[0], 1, actions.shape[-1])], dim=1)
        dataset = torch.cat([actions, obs.clone()], dim=-1)

        n_tp_data = dataset[:].shape[1] - 1
        time_steps = torch.arange(start=0, end=n_tp_data, step=1).float()
        time_steps = time_steps / len(time_steps)

        def collate_fn(batch, time_steps):
            batch = torch.stack(batch)
            return {
                's': batch[:, :-1, action_dim:],
                'a': batch[:, :-1, :action_dim],
                'y': batch[:, 1:, action_dim:],
                't': time_steps.unsqueeze(0).repeat(batch.shape[0], 1)
            }
        
        train_y, val_y, test_y = utils.split_train_val_test(dataset)
        self.dltrain = DataLoader(train_y, batch_size=100, shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, time_steps))
        self.dlval = DataLoader(val_y, batch_size=100, shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, time_steps))
        self.dltest = DataLoader(test_y, batch_size=100, shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, time_steps))
        
        self.model = model
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, lr_decay_step, lr_dacay)
        self.criterion = torch.nn.MSELoss()

    def train(self):
        waiting = 0
        best_loss = float('inf')
        for epoch in range(1000):
            iteration = 0

            self.model.train()
            for batch_dict in self.dltrain:
                self.optim.zero_grad()
                pred_y = self.model(batch_dict['s'], batch_dict['a'], batch_dict['t'])
                train_loss = self.criterion(pred_y, batch_dict['y'])
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optim.step()

                print(f'[epoch={epoch+1:04d}|iter={iteration+1:04d}] train_loss={train_loss:.5f} waiting={waiting}')
                self.plotter.log_loss('train_loss', train_loss.item())
                iteration += 1
                self.plotter.step()

            # validation
            self.model.eval()
            val_loss = self.val()
            self.plotter.log_loss('val_loss', val_loss.item())

            if val_loss < best_loss:
                print(f'best validation loss achieved at epoch {epoch + 1}: {val_loss}')
                best_loss = val_loss
                torch.save(self.model.state_dict(), 'saved_models/best_model.pt')
                waiting = 0
            elif waiting > self.patience:
                break
            else:
                waiting += 1

            torch.save(self.model.state_dict(), 'saved_models/final_model.pt')

    def val(self):
        n_batches = 0
        mse = 0
        for batch_dict in self.dlval:
            pred_y = self.model(batch_dict['s'], batch_dict['a'], batch_dict['t'])
            val_loss = self.criterion(pred_y, batch_dict['y'])
            mse += val_loss
            n_batches += 1
        mse /= n_batches
        return mse


    def finish(self, save='test.gif'):
        env = suite.load('hopper', 'stand')

        physics = env.physics

        model_path = 'saved_models/best_model.pt'
        if not os.path.exists(model_path):
            print('model path does not exist')
            return

        self.model.load_state_dict(torch.load(model_path))
        batch_dict = next(iter(self.dltest))
        
        pred_y = self.model(batch_dict['s'], batch_dict['a'], batch_dict['t'])

        pred_y = pred_y.detach().cpu().numpy()[0]

        #inv normalize
        pred_y[:, 3:self.qpos_dim] *= self.jnt_range[:, 1] - self.jnt_range[:, 0]
        pred_y[:, 3:self.qpos_dim] += self.jnt_range[:, 0]


        actual_y = batch_dict['y'].detach().cpu().numpy()[0]

        actual_y[:, 3:self.qpos_dim] *= self.jnt_range[:, 1] - self.jnt_range[:, 0]
        actual_y[:, 3:self.qpos_dim] += self.jnt_range[:, 0]

        frames_actual = []
        env.reset()

        pred_y[:, :3] = env.physics.data.qpos[:3]
        actual_y[:, :3] = env.physics.data.qpos[:3]

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
        generate_gif(frames, name=save)
        print('video saved')