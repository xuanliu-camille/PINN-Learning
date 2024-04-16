"""
this is a simple case of ODE 1
dx/dt = -x
x(0) = 0
"""

import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt

trange = np.linspace(0,10,101)
X0 = [1+0.01*i for i in range(100)]
data = []
# actual value
for x0 in X0:
    for t in trange:
        data.append([x0,t,(np.e**(-1*t))])

data = np.array(data)   # [x step,t step,x_real]
t_test = tf.constant(random.uniform(0,2))
with tf.GradientTape() as g:
    g.watch(t_test)
    x_test = tf.math.exp(-1 * t_test)
# what is this ?
print(g.gradient(x_test,x_test) + x_test)

# training setting
batch_size = 64
input_train = data[:,0:2] # [x step, t step]
output_train = data[:,2] # [x value]
# 组合input and output data
train_dataset = tf.data.Dataset.from_tensor_slices((input_train,output_train))
train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(batch_size)

# network
inputs = keras.Input(shape = (2,),name = 'X0_t')
x1 = layers.Dense(64, activation = 'relu')(inputs)
x2 = layers.Dense(64, activation = 'relu')(x1)
outputs = layers.Dense(1, name = 'predictions')(x2)
model = keras.Model(inputs = inputs, outputs = outputs)
optimizer = keras.optimizers.Adam(learning_rate = 1e-3)
# loss function
loss_fn = keras.losses.MeanSquaredError()
epochs = 100

for epoch in range(epochs):
    print('\nStart of epoch %d' %(epoch,))
    for step,(x_batch_train,y_batch_train) in enumerate(train_dataset):
        # print(x_batch_train)
        with tf.GradientTape(persistent=True) as tape: # training
            logits = model(x_batch_train,training = True)
            loss_value = loss_fn(y_batch_train,logits) # loss(xreal, xpredict)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x_batch_train)
                Y_hat = model(x_batch_train) # x predict
            dx_dt = tf.reduce_mean(tape2.gradient(Y_hat,x_batch_train),axis=0)[1] # dx/dt?
            x = tf.cast(tf.reduce_mean(Y_hat),dtype='float64')
            error = dx_dt + x
            l2 = tf.square(error)
        g2 = tape.gradient(l2,model.trainable_weights)
        grads = tape.gradient(loss_value,model.trainable_weights) + g2  # 用标签值训练
        optimizer.apply_gradients(zip(grads,model.trainable_weights))
    print('epoch%s'%epoch,error)
x0 = 1.5
x = []
for t in trange:
    x.append(model.predict(np.array([[x0,t]]))[0][0]) # 输入一对点进行预测的

plt.figure()
plt.plot(trange,x)
plt.plot(trange,[(np.e**(-1*t)) for t in trange])
plt.show()


