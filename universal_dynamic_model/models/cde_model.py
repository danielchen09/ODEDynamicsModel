import torch
from torch import nn
import torchcde

class CDEModel(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim):
        super().__init__()
        # [B, 1, S + A + 1](s0, a0, t0) -enc-> [B, 1, H](z0) -f-> [B, H, A + 1] -odesolve-> [B, T, H] -dec-> [B, T, S]
        x_dim = action_dim + 1
        self.encoder = nn.Linear(state_dim + action_dim + 1, latent_dim) # action + time
        class F(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, latent_dim * x_dim)
                )
                
            def forward(self, t, z):
                return self.net(z).view(-1, latent_dim, x_dim)
        self.f = F()
        self.decoder = nn.Linear(latent_dim, state_dim)
    
    def forward(self, s, a, t):
        s0 = s[:, 0, :]
        a0 = a[:, 0, :]
        t0 = t[:, 0, None]
        
        x = torch.cat([t.unsqueeze(-1), a], dim=-1)
        coeffs = torchcde.natural_cubic_spline_coeffs(x)
        X = torchcde.CubicSpline(coeffs)
        z0 = self.encoder(torch.cat([s0, a0, t0], dim=-1))
        z = torchcde.cdeint(X=X, func=self.f, z0=z0, t=t[0])
        breakpoint()
        return self.decoder(z)

if __name__ == '__main__':
    s = torch.randn(1000, 100, 17)
    a = torch.randn(1000, 100, 17)
    t = torch.linspace(0, 1, 100).unsqueeze(0).repeat(1000, 1)
    model = CDEModel(s.shape[-1], a.shape[-1], 128)
    print(model(s, a, t).shape)