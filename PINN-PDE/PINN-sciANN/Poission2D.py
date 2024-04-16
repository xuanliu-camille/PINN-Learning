import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
from sciann.utils.math import diff,sign,sin,pi
import tensorflow as tf
import torch
from tensorflow import keras
# import sciann.math.sin as sin

# setting variables and functions
x = sn.Variable('x')
y = sn.Variable('y')
u = sn.Functional('u',[x,y],8*[30],'tanh')

# setting loss functions
L1 = diff(u,x,order=2) + diff(u,y,order=2) + 2 * pi**2 * sin(pi*x) * sin(pi*y)
Xmin = 0
Xmax = 1
Ymin = 0
Ymax = 1
C1 = (1+sign(x-Xmax)) * (u-sin(pi*x)*sin(pi*y))
C2 = (1-sign(x-Xmin)) * (u-sin(pi*x)*sin(pi*y))
C3 = (1+sign(y-Ymax)) * (u-sin(pi*x)*sin(pi*y))
C4 = (1-sign(y-Ymin)) * (u-sin(pi*x)*sin(pi*y))
m = sn.SciModel([x,y],[L1,C1,C2,C3,C4],'mse','adam')
# m = sn.SciModel([x,y],[L1,u],'mse','Adam')
#
a = Xmax - Xmin
b = Ymax - Ymin
n = 100
x_data,y_data = np.meshgrid(np.linspace(0,a,n),
                            np.linspace(0,b,n))
#
# # training
h = m.train([x_data,y_data],5*['zeros'],learning_rate=0.002,epochs=1500,verbose=0)

# path = './poission_weights'
# m.save(path)

# m = keras.models.load_model('poission_weights.index')


x_test, y_test = np.meshgrid(
    np.linspace(0,a,n),
    np.linspace(0,b,n))
# np.array
u_pred = u.eval(m, [x_test, y_test])


# type:tensor
# u_real = tf.sin(pi * x_test) * tf.sin(pi * y_test)
u_real = np.sin(np.pi * x_test) * np.sin(np.pi * y_test)
# sess = tf.Session()
# u_real = ureal.eval(session=sess)
# print(type(u_real))

error = u_pred - u_real

plt.subplot(131)
plt.pcolor(x_test, y_test, u_real, cmap='seismic')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.title('real result')

plt.subplot(132)
plt.pcolor(x_test, y_test, u_pred, cmap='seismic')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.title('predicted result')

plt.subplot(133)
plt.pcolor(x_test, y_test, error, cmap='seismic')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.title('error')
plt.show()
