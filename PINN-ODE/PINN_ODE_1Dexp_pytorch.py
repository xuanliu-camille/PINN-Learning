"""
this is a simple case of ODE 1
dx/dt = -x
x(0) = 1
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## setting network
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # setting hidden layers and number of neurons, num_of_input: 2 (x and t)
        self.hidden_layer1 = nn.Linear(2, 64)
        self.hidden_layer2 = nn.Linear(64, 64)
        self.hidden_layer3 = nn.Linear(64, 64)
        # setting output layer, num_of_output:1 (u)
        self.output_layer = nn.Linear(64,1)

    # calculating forward calculation result
    def forward(self,x,t):
        inputs = torch.cat([x,t],axis = 1) # cat: combining 2 arrays of 1 col to 1 array of 2 cols
        # activation function: sigmoid
        layer1_out = torch.relu(self.hidden_layer1(inputs))
        layer2_out = torch.relu(self.hidden_layer2(layer1_out))
        layer3_out = torch.relu(self.hidden_layer3(layer2_out))
        output = self.output_layer(layer3_out)
        return output

## setting PINN model
net = Net()
net = net.to(device)
mse_cose_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())

# residual function
def f(x,t,net):
    x = net(x,t)
    x_t = torch.autograd.grad(x.sum(),t,create_graph=True)[0]
    ode = x_t + x
    return ode

# label
trange = np.linspace(0,10,101)
xrange = np.linspace(1,2,101)
xmesh,tmesh = np.meshgrid(xrange,trange)
t_label = tmesh.reshape(-1,1)
x_0 = xmesh.reshape(-1,1)
x_label = np.e**(-1 * t_label)
# x_t0 = 0

## training
iterations = 100*32
previous_validation_loss = 99999999.0
loss_list = []

# train with given posint
t_train = np.linspace(0,10,101).reshape(-1,1)
x_train = np.linspace(1,2,101).reshape(-1,1)
for epoch in range(iterations):
    # forcing gradient to zeros
    optimizer.zero_grad()

    # loss: initial condition
    pt_t_IC = Variable(torch.from_numpy(t_label).float(),requires_grad = False).to(device)
    pt_x_IC = Variable(torch.from_numpy(x_0).float(),requires_grad = False).to(device)
    net_IC_out = net(pt_x_IC,pt_t_IC)
    pt_x_label = Variable(torch.from_numpy(x_label).float(),requires_grad = False).to(device)
    mse_IC = mse_cose_function(net_IC_out,pt_x_label)

    # loss: based on ODE
    x_collocation = x_train
    t_collocation = t_train
    all_zeros = np.zeros(t_collocation.shape)
    pt_x = Variable(torch.from_numpy(x_collocation).float(),requires_grad = True).to(device)
    pt_t = Variable(torch.from_numpy(t_collocation).float(),requires_grad = True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(),requires_grad = True).to(device)
    f_out_coll = f(pt_x,pt_t,net)
    mse_f_coll = mse_cose_function(f_out_coll,pt_all_zeros)

    # loss = mse_IC + mse_f_coll
    loss = mse_f_coll

    loss.backward()
    optimizer.step()

    with torch.autograd.no_grad():
        print(epoch,'Training Loss: ',loss.data)

## testing
x_test = 1.5
t_test = trange.reshape(-1,1)
x_test = 1.5 * np.ones(t_test.shape)
pt_x = Variable(torch.from_numpy(x_test).float(), requires_grad=True).to(device)
pt_t = Variable(torch.from_numpy(t_test).float(), requires_grad=True).to(device)
x_pred = net(pt_x,pt_t)
x_pred = x_pred.data.cpu().numpy()
x_real = np.e**(-1*t_test)
error = x_pred - x_real

plt.figure()
plt.plot(t_test,x_pred,label = 'prediction')
plt.plot(t_test,x_real,label = 'real')
plt.plot(t_test,error,label = 'error')
plt.legend()
plt.show()



