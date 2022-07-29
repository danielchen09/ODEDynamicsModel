from torch import nn
import torch
import torch.functional as F
import nfe.experiments.latent_ode.lib.utils as utils
from nfe.experiments.latent_ode.lib.create_latent_ode_model import SolverWrapper
from nfe.models import CouplingFlow, GRUFlow, ContinuousGRULayer


class LinearEncoder(nn.Module):
    """
    input -> z0
    """
    def __init__(self, input_dim, latent_dim, z0_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.transform_z0 = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Linear(latent_dim, 100),
            nn.Tanh(),
            nn.Linear(100, z0_dim * 2)
        )
        utils.init_network_weights(self.transform_z0)
    
    def forward(self, data, time_steps, run_backwards=True, save_info=False):
        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())
        # breakpoint()
        n_traj, n_tp, n_dims = data.size()
        data = data[:, 0, :] # (n_traj, n_dims) x0's
        data = data.reshape(1, n_traj, n_dims)
        mean_z0, std_z0 = self.transform_z0(data).chunk(2, dim=-1)
        std_z0 = F.softplus(std_z0)
        
        return utils.sample_standard_gaussian(mean_z0, std_z0)


class ODERNN(nn.Module):
    def __init__(self, 
                 state_dim,
                 action_dim, 
                 hidden_dim, 
                 num_layers,
                 flow_layers,
                 time_net,
                 time_hidden_dim,
                 flow_model='gru',
                 mode='truth'):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mode = mode # auto, truth, s0

        if flow_model == 'gru':
            self.diffeq_solver = GRUFlow(
                hidden_dim,
                flow_layers,
                time_net,
                time_hidden_dim
            ) # s, a, t -> s', a' (ignore)
        elif flow_model == 'coupling':
            self.diffeq_solver = CouplingFlow(
                hidden_dim,
                flow_layers,
                [128, 128],
                time_net,
                time_hidden_dim
            )
        self.gru = nn.GRU(state_dim + action_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, state_dim)

    def forward(self, s, a, t, h=None):
        # x: [B, T0:Tn-1, input_dim]
        # t: [B, T1:Tn]
        # h: [B, num_layers, hidden_dim]

        n_batch = s.shape[0]
        n_ts = s.shape[1]
        device = s.device

        if h is None:
            h = torch.zeros(n_batch, self.num_layers, self.hidden_dim)
        h.to(device)
        hiddens = torch.zeros(n_batch, n_ts, self.hidden_dim).to(device)
        outputs = torch.zeros(n_batch, n_ts, self.state_dim).to(device)

        s0 = s[:, 0, None]
        last_output = s0
        for i in range(t.shape[1]):
            h = self.diffeq_solver(h, t[:, i:i+1, None])
            hiddens[:, i, None] = h

            if self.mode == 'auto':
                gru_in = torch.cat([last_output, a[:, i, None]], dim=-1)
            elif self.mode == 'truth':
                gru_in = torch.cat([s[:, i, None], a[:, i, None]], dim=-1)
            elif self.mode == 's0':
                gru_in = torch.cat([s0, a[:, i, None]], dim=-1)

            _, h = self.gru(gru_in, h.transpose(0, 1))
            h = h.transpose(0, 1)
            o = self.decoder(h)
            last_output = o
            outputs[:, i, None] = o

        return outputs, hiddens


class FlowModel(nn.Module):
    def __init__(self, 
                 state_dim,
                 action_dim, 
                 latent_dim,
                 flow_layers,
                 time_net,
                 time_hidden_dim,
                 flow_model='gru',
                 action_emb_dim=-1,
                 mode='auto'):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.embed_action = False
        if action_emb_dim != -1:
            self.embed_action = True
            self.action_encoder = nn.Linear(action_dim, action_emb_dim)
        else:
            action_emb_dim = action_dim

        self.encoder = nn.Linear(state_dim, latent_dim)

        # ode part
        self.odernn = ODERNN(state_dim, action_emb_dim, latent_dim, 1, flow_layers, time_net, time_hidden_dim, flow_model=flow_model, mode=mode)
        # self.decoder = nn.Linear(latent_dim, state_dim)

    # [B, T0:Tn-1, S+A] -> [B, T1:Tn, S]
    def forward(self, s, a, t):
        s0 = s[:, 0, None]
        h0 = self.encoder(s0)
        if self.embed_action:
            a = self.action_encoder(a)
        outputs, hiddens = self.odernn(s, a, t, h0)
        return outputs
    
if __name__ == '__main__':
    s = torch.randn(1000, 100, 17)
    a = torch.randn(1000, 100, 17)
    t = torch.randn(1000, 100)
    model = FlowModel(s.shape[-1], a.shape[-1], 256, 2, 'TimeTanh', 256)
    print(model(s, a, t).shape)