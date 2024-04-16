"""
du/dt = cos(2pit)
u(0) = 1
exact solution: u(t) = 1/(2pi) * sin(2pi*t) + 1
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import matplotlib.pyplot as plt
import math
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## define network
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # setting hidden layers and number of neurons, num_of_input: 2 (x and t)
        self.hidden_layer1 = nn.Linear(1, 32)
        self.hidden_layer2 = nn.Linear(32, 32)
        # setting output layer, num_of_output:1 (u)
        self.output_layer = nn.Linear(32,1)

    # calculating forward calculation result
    def forward(self,t):
        inputs = torch.cat([t],axis = 1) # cat: combining 2 arrays of 1 col to 1 array of 2 cols
        # activation function: sigmoid
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        output = self.output_layer(layer2_out)
        return output

## setting PINN model
net = Net()
net = net.to(device)
mse_cost_function = torch.nn.MSELoss() # applying MSE as loss function, mean squared error
optimizer = torch.optim.Adam(net.parameters())

## define ODE system
def f(t,net):
    u = net(t)
    u_t = torch.autograd.grad(u.sum(),t,create_graph=True)[0]
    ode = u_t - torch.cos(2 * math.pi * t)
    return ode

## training
t_train = (np.random.rand(30)*2).reshape(-1,1)

# initial condition
t0 = np.zeros((1,1)).reshape(-1,1)
u0 = np.ones((1,1)).reshape(-1,1)

iterations = 8000
for epoch in range(iterations):
    optimizer.zero_grad()

    # IC loss
    pt_t_IC = Variable(torch.from_numpy(t0).float(), requires_grad = False).to(device)
    pt_u_IC = Variable(torch.from_numpy(u0).float(), requires_grad = False).to(device)
    net_out_IC = net(pt_t_IC)
    mse_IC = mse_cost_function(net_out_IC,pt_u_IC)

    # ODE loss
    t_collocation = t_train
    all_zeros = np.zeros(t_collocation.shape)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad = True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad = True).to(device)
    f_out_coll = f(pt_t_collocation,net)
    mse_f_coll = mse_cost_function(f_out_coll,pt_all_zeros)

    loss = mse_IC + mse_f_coll
    loss.backward()
    optimizer.step()

    with torch.autograd.no_grad():
        print(epoch,'Training Loss:', loss.data)

## prediction
t_test = np.linspace(0,2,100).reshape(-1,1)
u_real = (1/(2 * np.pi)) * np.sin((2 * np.pi) * t_test) + 1
t_test_tensor = Variable(torch.from_numpy(t_test).float(), requires_grad=True).to(device)
u_pred_tensor = net(t_test_tensor)
u_pred = u_pred_tensor.data.cpu().numpy()
error = u_real - u_pred

plt.figure()
plt.plot(t_test, u_pred, label = 'prediction')
plt.plot(t_test, u_real, label = 'real')
plt.plot(t_test, error, label = 'error')
plt.legend()
plt.show()