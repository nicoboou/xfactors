import torch
import torch.nn.functional as F
import torch.nn as nn

def kl(mu: torch.Tensor, logvar: torch.Tensor, by_latent: bool=False) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=-20.0, max=20.0)
    if by_latent:
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)
    else:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

def mse(x_hat_logits: torch.Tensor, x: torch.Tensor) :
    return F.mse_loss(x_hat_logits, x, reduction="mean")

def cross_cov(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a = a - a.mean(dim=0)    
    # b = b - b.mean(dim=0)

    # cov = (a.T @ b) / (a.size(0)-1)       
    # return cov/(a.std()*b.std())     
    n = a.shape[0]
    X_c = a - a.mean(dim=0, keepdim=True)
    Y_c = b - b.mean(dim=0, keepdim=True)

    cov = X_c.T @ Y_c / (n - 1)

    std_X = a.std(dim=0, unbiased=True).unsqueeze(1)  # [d, 1]
    std_Y = b.std(dim=0, unbiased=True).unsqueeze(0)  # [1, p]

    corr = cov / (std_X @ std_Y + 1e-8)  # [d, p]
    return corr


def decorrelate_params(mu_s, logvar_s, mu_t, logvar_t,):
    return torch.clip(torch.norm(cross_cov(mu_s, mu_t), p="fro"), min=-1, max=1)

    
#InfoNCE supervised
class InfoNCESupervised(nn.Module) :
    def __init__(self, temperature: float=0.07, eps: float=1e-8) :
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z: torch.Tensor,
                    labels: torch.Tensor,) -> torch.Tensor:
  
        device = z.device
        batch_size = z.size(0)
        z = F.normalize(z, dim=1)
        assert not torch.isnan(z).any(), "In NCE, z is Nan"

        sim = torch.matmul(z, z.t()) / self.temperature
        assert not torch.isnan(z).any(), "sim is Nan"
        sim = torch.clamp(sim, min=-100, max=100)
        assert not torch.isnan(sim).any(), "sim clamped is Nan"
        mask_self = torch.eye(batch_size, device=device).bool()
        sim.masked_fill_(mask_self, float("-inf"))

        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.t()) & ~mask_self  # (B, B)

        valid = mask_pos.any(dim=1)
        if not torch.any(valid):
            return sim.new_tensor(0.0, requires_grad=True)
        
        # exp_sim = torch.exp(sim)
        # denom = exp_sim.sum(dim=1) 
        # numer = (exp_sim * mask_pos.float()).sum(dim=1)

        # loss = -torch.log((numer + self.eps) / (denom + self.eps))


        sim = sim[valid]              
        mask_pos = mask_pos[valid]

        log_denom = torch.logsumexp(sim, dim=1)  
        assert not torch.isnan(log_denom).any(), "log_denom is nan"
        sim_pos = sim.masked_fill(~mask_pos, float("-inf"))
        log_numer = torch.logsumexp(sim_pos, dim=1) 
        assert not torch.isnan(log_numer).any(), "log_numer is nan"

        loss = -(log_numer - log_denom) 
        return torch.clip(loss.mean(), max=1e5)

        #return loss.mean()