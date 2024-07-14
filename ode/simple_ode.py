import numpy as np

from kan import KAN, LBFGS
import torch
import matplotlib.pyplot as plt
from torch import autograd
from tqdm import tqdm

# We aim to solve 1D ordinary differential equation (ODE) with initial condition
# dy/dx = exp(x), y(0) = 1

dim = 1
np_i = 100  # number of interior points (along each dimension)
ranges = [0, 2]

model = KAN(width=[1, 2, 1], grid=5, k=3, grid_eps=1.0)


# solve the following ODE dy/dx = (x-y)/(x-2*y) with y(0) = 1
# define solution
# source_fun = lambda x: torch.exp(x[:, 0])


def sol_fun(x, y):
    return (x - y) / (x - 2 * y)


# range
x_i = torch.linspace(ranges[0], ranges[1], np_i).view(-1, 1)

# initial condition
helper = torch.tensor([0.0])
x_b = torch.stack([helper, torch.zeros_like(helper)]).permute(1, 0)

steps = 20
alpha = 0.1
log = 1


def train():
    optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32,
                      tolerance_change=1e-32, tolerance_ys=1e-32)

    pbar = tqdm(range(steps), desc='description')

    for _ in pbar:
        def closure():
            global ode_loss, ic_loss
            optimizer.zero_grad()
            # interior loss
            sol_pred = model(x_i)
            sol = sol_fun(x_i, torch.zeros_like(x_i))
            ode_loss = torch.mean((sol - sol_pred) ** 2)



            # initial condition loss
            ic_true = torch.exp(helper)
            ic_pred = model(torch.stack([helper, torch.zeros_like(helper)]).permute(1, 0))
            ic_loss = torch.mean((ic_true - ic_pred) ** 2)

            loss = alpha * ode_loss + ic_loss
            loss.backward()
            return loss

        if _ % 5 == 0 and _ < 50:
            model.update_grid_from_samples(x_i)

        optimizer.step(closure)
        sol = sol_fun(x_i, torch.zeros_like(x_i))
        loss = alpha * ode_loss + ic_loss
        l2 = torch.mean((model(x_i) - sol) ** 2)

        if _ % log == 0:
            pbar.set_description("ode loss: %.2e | ic loss: %.2e | l2: %.2e " % (
                ode_loss.cpu().detach().numpy(), ic_loss.cpu().detach().numpy(), l2.detach().numpy()))


# train
train()

# plot
model.plot(beta=10)
plt.show()

# predict
x = torch.linspace(ranges[0], ranges[1], 100).view(-1, 1)
sol = sol_fun(x)
sol_pred = model(x)
l2 = torch.mean((sol - sol_pred) ** 2)
print('l2 error: ', l2)

# plot
plt.plot(x.detach().numpy(), sol.detach().numpy(), label='true', linestyle='--', color='black')
plt.plot(x.detach().numpy(), sol_pred.detach().numpy(), label='pred')
plt.legend()
plt.show()
