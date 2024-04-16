"""
1D problem: du/dx = 2du/dt + u, u(x,t=0) = 6e^(-3x)
input: x,t, x < [0,2], t < [0,1]
output: u, u(x,t) = 6e^(-3x-2t)
setting f = du/dx - 2du/dt - u
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### setting network
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # setting hidden layers and number of neurons, num_of_input: 2 (x and t)
        self.hidden_layer1 = nn.Linear(2, 5)
        self.hidden_layer2 = nn.Linear(5, 5)
        self.hidden_layer3 = nn.Linear(5, 5)
        self.hidden_layer4 = nn.Linear(5, 5)
        self.hidden_layer5 = nn.Linear(5, 5)
        # setting output layer, num_of_output:1 (u)
        self.output_layer = nn.Linear(5,1)

    # calculating forward calculation result
    def forward(self,x,t):
        inputs = torch.cat([x,t],axis = 1) # cat: combining 2 arrays of 1 col to 1 array of 2 cols
        # activation function: sigmoid
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out)
        return output

### setting PINN model
net = Net()
net = net.to(device)
mse_cost_function = torch.nn.MSELoss() # applying MSE as loss function, mean squared error
optimizer = torch.optim.Adam(net.parameters())

# PDE as loss function, f function
def f(x,t,net):
    u = net(x,t)
    u_x = torch.autograd.grad(u.sum(),x,create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(),t,create_graph=True)[0]
    pde = u_x - 2 * u_t - u
    return pde

# Boundary conditions (BC)
# sample 500 points on x, ranging (0,2)
x_bc = np.random.uniform(low = 0.0, high = 2.0, size = (500,1))
t_bc = np.zeros((500,1))
# calculating u on boundary, u(x,0)=6e^(-3x)
u_bc = 6 * np.exp(-3 * x_bc)

### training
iterations = 20000
previous_validation_loss = 99999999.0 # what is this for?
for epoch in range(iterations):
    # forcing gradient to zeros
    optimizer.zero_grad()

    # loss based on boundary conditions
    pt_x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad = False).to(device)
    pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad = False).to(device)
    pt_u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad = False).to(device)
    net_bc_out = net(pt_x_bc,pt_t_bc) # output of u(x,t)
    mse_u = mse_cost_function(net_bc_out,pt_u_bc) # mse between u value from given equation and net calculation result

    # loss based on PDE
    x_collocation = np.random.uniform(low = 0.0, high = 2.0, size = (500,1))
    t_collocation = np.random.uniform(low = 0.0, high = 1.0, size = (500,1))
    all_zeros = np.zeros((500,1))
    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=True).to(device)
    f_out = f(pt_x_collocation,pt_t_collocation,net)
    mse_f = mse_cost_function(f_out,pt_all_zeros)

    # combining 2 parts of loss function
    loss = mse_u + mse_f

    loss.backward() # calculate gradients using backward propagation
    optimizer.step() # equivalent to theta_new = theta_old - alpha * derivative

    with torch.autograd.no_grad():
        print(epoch,'Training Loss: ', loss.data)

### plotting results
fig = plt.figure()
# ax = fig.add_axes(Axes3D(fig))
ax = fig.add_subplot(121,projection='3d')

x=np.arange(0,2,0.02)
t=np.arange(0,1,0.02)
ms_x, ms_t = np.meshgrid(x, t)
x = np.ravel(ms_x).reshape(-1,1)
t = np.ravel(ms_t).reshape(-1,1) # reshape meshgrid

pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)
pt_u = net(pt_x,pt_t)
u = pt_u.data.cpu().numpy() # data: generate a new tensor, numpy: transfer tensor to ndarray
ms_u = u.reshape(ms_x.shape)

# plotting prediction result
surf = ax.plot_surface(ms_x,ms_t,ms_u, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()

# plotting error
# fig2 = plt.figure()
ax2 = fig.add_subplot(122,projection='3d')

rl_u = 6 * np.exp(-3 * x - 2 * t) # numpy calculation
rl_u = rl_u.reshape(ms_x.shape)
error = ms_u - rl_u

surf_error = ax2.plot_surface(ms_x,ms_t,error, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax2.zaxis.set_major_locator(LinearLocator(10))
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

### saving model
torch.save(net.state_dict(), "model_uxt.pt")






