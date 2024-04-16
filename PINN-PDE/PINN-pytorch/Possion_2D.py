"""
this script is for the study case of Poission 2D
-d2u/dx2 - d2u/dy2 = 2 * pi**2 * sin(pi*x) * sin(pi*y)
u = sin(pi * x) * sin(pi * y) on boundaries

input x,y, x,y < [0,1]
output u, u(x,y) = sin(pi * x) * sin(pi * y)
setting f = d2u/dx2 + d2u/dy2 + 2 * pi**2 * sin(pi*x) * sin(pi*y)
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
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
    def forward(self,x,y):
        inputs = torch.cat([x,y],axis = 1) # cat: combining 2 arrays of 1 col to 1 array of 2 cols
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
def f(x,y,net):
    u = net(x,y)
    u_x = torch.autograd.grad(u.sum(),x,create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(),x,create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(),y,create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(),y,create_graph=True)[0]
    pde = u_xx + u_yy + 2 * (math.pi)**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)
    return pde

# Boundary conditions (BC)
# generate boundary conditions, n_bc: number of boundaries, n_data_perbc: number of point on each boundary
def generate_bc(n_bc,n_data_perbc,x_lb,x_up,y_lb,y_up):
    data = np.zeros([n_bc,n_data_perbc,3]) # x, y and u value
    # generate points on each boundary, left, right, down and up
    for i,j in zip(range(n_bc),[x_lb,x_up,y_lb,y_up]):
        points_x = np.random.uniform(low = x_lb, high = x_up, size = (1,n_data_perbc))
        points_y = np.random.uniform(low = y_lb, high = y_up, size = (1,n_data_perbc))
        if i < 2:
            data[i, :, 0] = j
            data[i, :, 1] = points_x
        else:
            data[i, :, 0] = points_y
            data[i, :, 1] = j
    # calculate u value on boundaries
    for i in range(n_bc):
        data[i, :, 2] = np.sin(np.pi * data[i, :, 0] * np.sin(np.pi * data[i, :, 1]))
    # expand matrix to array (n,1)
    data = data.reshape(n_data_perbc * n_bc, 3)
    x_bc, y_bc, u_bc = map(lambda x: np.expand_dims(x, axis=1),
                           [data[:, 0], data[:, 1], data[:, 2]])
    return x_bc,y_bc,u_bc

x_lb = 0
x_up = 1
y_lb = 0
y_up = 1
x_bc,y_bc,u_bc = generate_bc(4,25,x_lb,x_up,y_lb,y_up)


### training
iterations = 1500
previous_validation_loss = 99999999.0
for epoch in range(iterations):
    # forcing gradient to zeros
    optimizer.zero_grad()

    # loss based on boundary conditions
    pt_x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad = False).to(device)
    pt_y_bc = Variable(torch.from_numpy(y_bc).float(), requires_grad = False).to(device)
    pt_u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad = False).to(device)
    net_bc_out = net(pt_x_bc,pt_y_bc) # output of u(x,t)
    mse_u = mse_cost_function(net_bc_out,pt_u_bc) # mse between u value from given equation and net calculation result

    # loss based on PDE
    n_points = 100
    np_x_coll = np.random.uniform(low = x_lb, high = x_up, size = (1,n_points))
    np_y_coll = np.random.uniform(low = y_lb, high = y_up, size = (1,n_points))
    ms_x_coll, ms_y_coll = np.meshgrid(np_x_coll, np_y_coll)
    x_collocation = np.ravel(ms_x_coll).reshape(-1, 1)
    y_collocation = np.ravel(ms_y_coll).reshape(-1, 1)
    all_zeros = np.zeros((n_points * n_points,1))

    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(y_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=True).to(device)
    f_out = f(pt_x_collocation,pt_t_collocation,net)
    mse_f = mse_cost_function(f_out,pt_all_zeros)

    # combining 2 parts of loss function
    loss = mse_u + mse_f

    loss.backward() # calculate gradients using backward propagation
    optimizer.step() # equivalent to theta_new = theta_old - alpha * derivative

    with torch.autograd.no_grad():
        print(epoch,'Training Loss: ', loss.data)

### plotting
fig = plt.figure()
x = np.arange(0,1,0.01)
y = np.arange(0,1,0.01)

# predicted u value
ms_x, ms_y = np.meshgrid(x, y)
x = np.ravel(ms_x).reshape(-1,1)
y = np.ravel(ms_y).reshape(-1,1)

pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True).to(device)
pt_u = net(pt_x,pt_y)
u = pt_u.data.cpu().numpy() # data: generate a new tensor, numpy: transfer tensor to ndarray
ms_upd = u.reshape(ms_x.shape)

plt.subplot(131)
plt.pcolor(ms_x, ms_y, ms_upd, cmap='seismic')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.title('predicted u')

# real u value
rl_u = np.sin(np.pi * x) * np.sin(np.pi * y)
ms_url = rl_u.reshape(ms_x.shape)

plt.subplot(132)
plt.pcolor(ms_x, ms_y, ms_url, cmap='seismic')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.title('real u')

# error
ms_uer = ms_upd - ms_url

plt.subplot(133)
plt.pcolor(ms_x, ms_y, ms_uer, cmap='seismic')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.title('error')
plt.show()

fig.colorbar()
plt.show()





