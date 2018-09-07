"""PyTorch code for a one layer neural network."""

import torch

device = 'cuda:0'
N, D = 3, 2

x = torch.randn(N, D, device=device)
y = torch.randn(N, 1, device=device)
w = torch.randn(D, 1, requires_grad=True,
                device=device)

res = torch.matmul(x, w)
loss = 0.5 * torch.sum(torch.pow(res - y, 2))

loss.backward()
print(w.grad)
