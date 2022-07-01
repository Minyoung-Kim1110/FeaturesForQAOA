''' 
Author: Jaehwan Kim ( Jul. 1, 2022)
2022 Hackaton 
Team: ThankQ
description: 
    Generate Neural Network (dense net) with QAOA parameter betas and gammas
''' 
import torch
import torch.nn as nn

class DenseNet(nn.Module):
    def __init__(self, data: torch.Tensor, n_wires: int = 5, n_layers: int = 1):
        super(DenseNet, self).__init__()
        self.data = torch.Tensor(data)
        self.betas = nn.Parameter(torch.rand(n_layers).to(torch.float64) * 0.01, requires_grad=True)
        self.gammas = nn.Parameter(torch.rand(n_layers).to(torch.float64) * 0.01, requires_grad=True)
        # self.embed = nn.Embedding(num_embeddings=n_wires, embedding_dim=3).to(torch.float64)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )

    def forward(self, i: int):
        return self.linear_relu_stack(self.data[i])
