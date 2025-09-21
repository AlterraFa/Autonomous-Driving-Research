# %% Steering GMM
import torch

torch.random.manual_seed(45)

N = 3
batch = 5
GMM_param = torch.randn((batch, 3 * N))  # shape (batch, 3 * N)

weights, muy_weights, sigma_weights = torch.chunk(GMM_param, 3, 1)

weights = torch.softmax(weights, dim=1) 
muy = muy_weights                       
sigma = torch.exp(sigma_weights)

def gaussian(sample, muy, sigma):
    return (1 / (2 * torch.pi * sigma ** 2) ** .5) * torch.exp( -(sample - muy) ** 2 / (2 * sigma ** 2))

prob_per_mode = gaussian(.5, muy, sigma)
prob = torch.sum(weights * prob_per_mode, dim = 1) / N
print(prob_per_mode)
# %% Waypoint GMM
import torch
import math
torch.random.manual_seed(45)

N = 3; waypoints = 5
batch = 7
GMM_param = torch.randn((batch, N + N * waypoints * 2 + N * waypoints * 2))

weights, muy_weights, sigma_weights = torch.split(GMM_param, 
                                                  [N, N * waypoints * 2, N * waypoints * 2],
                                                  dim = 1)

weights = torch.softmax(weights, dim = 1).unsqueeze(-1)
muy = muy_weights.view(batch, N, waypoints, 2)
sigma = torch.exp(sigma_weights).view(batch, N, waypoints, 2) # (batch, modes, waypoints, dim)

def multivariate_gaussian(sample, mu, sigma):
    _, N, *_ = mu.shape

    sample = sample.unsqueeze(1).expand(-1, N, -1, -1)

    # joint probability distribution between x and y => norm const is prod while exp term is sum
    norm_const = (1.0 / (torch.sqrt(torch.tensor(2.0 * math.pi)) * sigma)).prod(dim = 3)
    exp_term = torch.exp(-0.5 * (((sample - mu) / sigma) ** 2).sum(dim = 3))

    return norm_const * exp_term

pred = torch.rand((batch, waypoints, 2))
prob_per_mode = multivariate_gaussian(pred, muy, sigma) * weights
print(prob_per_mode)
# %%
import numpy as np

data = np.load("./data/recording_20250905_231635_best_spatial/metadata/000000.npy", allow_pickle = True)
print(data)