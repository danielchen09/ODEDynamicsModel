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
                 input_dim, 
                 hidden_dim, 
                 num_layers,
                 flow_layers,
                 time_net,
                 time_hidden_dim,
                 flow_model='gru'):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

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
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, x, t, h=None):
        # x: [B, T0:Tn-1, input_dim]
        # t: [B, T1:Tn]
        # h: [B, num_layers, hidden_dim]

        if h is None:
            h = torch.zeros(x.shape[0], self.num_layers, self.hidden_dim)
        h.to(x)
        hiddens = torch.zeros(*x.shape[:-1], self.hidden_dim).to(x)

        for i in range(t.shape[1]):
            h = self.diffeq_solver(h, t[:, i:i+1, None])
            hiddens[:,i,None] = h

            _, h = self.gru(x[:,i,None], h.transpose(0, 1))
            h = h.transpose(0, 1)

        return hiddens


class FlowModel(nn.Module):
    def __init__(self, 
                 state_dim,
                 action_dim, 
                 latent_dim,
                 flow_layers,
                 time_net,
                 time_hidden_dim,
                 flow_model='gru',
                 action_emb_dim=-1):
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
        self.odernn = ODERNN(action_emb_dim + state_dim, latent_dim, 1, flow_layers, time_net, time_hidden_dim, flow_model=flow_model)
        self.decoder = nn.Linear(latent_dim, state_dim)

    # [B, T0:Tn-1, S+A] -> [B, T1:Tn, S]
    def forward(self, s, a, t):
        s0 = s[:, 0:1, :]
        h0 = self.encoder(s0)
        if self.embed_action:
            a = self.action_encoder(a)
        hiddens = self.odernn(torch.cat([a, s0.repeat(1, a.shape[1], 1)], dim=-1), t, h0)
        return self.decoder(hiddens)
    
if __name__ == '__main__':
    s = torch.randn(1000, 100, 17)
    a = torch.randn(1000, 100, 17)
    t = torch.randn(1000, 100)
    model = FlowModel(s.shape[-1], a.shape[-1], 100, 2, 'TimeLinear', 100)
    print(model(s, a, t).shape)