# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:59:30 2020

@author: Majid Mobini
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:40:31 2019

@author: Majid Mobini
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 01:33:35 2019

@author: Majid Mobini
"""
import inverse_utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.integrate import odeint
from ipywidgets import interactive
from IPython.display import display
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
#
N=1
angle=0.0
tmax=10000
sigma=10.0
beta=8./3
rho=28.0
t = np.arange(0.0, tmax, 1)

def two_lorenz_odes(X, t):
    x, y, z = X
    dx = sigma * (y - x)
    dy = -(x * z) + (rho*x) - y
    dz = (x * y )- beta*z

    return (dx, dy, dz)


y0 = [.1, .1, .1]
f = odeint(two_lorenz_odes, y0, t)
x, y, z = f.T  # unpack columns

    # choose a different color for each trajectory



plt.show()
plt.plot(x.T)
plt.show()
#####################################################

raw_data =x.T
sequences = 1
max_length = 500
gap = 0
sequence_lengths = np.empty(sequences, dtype=int)
train_lengths=sequence_lengths
test_lengths =sequence_lengths

#np.random.seed(0)

X= raw_data[0:10000]+ np.random.normal(0, .4, [10000])
Y= raw_data[0+gap:10000+gap] 
Z= raw_data[0:500]  + np.random.normal(0,0.5, [500])
W= raw_data[0+gap:500+gap]

X = inverse_utils.normalise(X) #normalise signal to range [-1, 1]
Y = inverse_utils.normalise(Y) #normalise signal to range [-1, 1]
Z = inverse_utils.normalise(Z) #normalise signal to range [-1, 1]
W = inverse_utils.normalise(W) #normalise signal to range [-1, 1]


train_X= X.reshape((sequences, 10000, 1))
train_Y= Y.reshape((sequences, 10000, 1))

test_X= Z.reshape((sequences, 500, 1))
test_Y= W.reshape((sequences, 500, 1))

# Basic definitions
N = 64  # Size of recurrent neural network
T = 10000  # Maximum length of training time series
n = 1  # Number of training sequences
n_test = 1 # Number of test sequences
m = 1  # Output dimension
d = 1  # Input dimension
epochs = 800  # Maximum mnumber of training epochs
learning_rate = 1e-4  # Learning rate


# Placeholders
inputs  = tf.placeholder(tf.float32, [None, None, d])
target = tf.placeholder(tf.float32, [None, None, m])
lengths = tf.placeholder(tf.int64)

# Network architecture RNN
cell = tf.nn.rnn_cell.GRUCell(N)
rnn_output, _ = tf.nn.dynamic_rnn(
    cell, inputs, sequence_length=lengths, dtype=tf.float32)

# Note the following reshaping:
#   We want a prediction for every time step.
#   Weights of fully connected layer should be the same (shared) for every time step.
#   This is achieved by flattening the first two dimensions.
#   Now all time steps look the same as individual inputs in a batch fed into a feed-forward network.
rnn_output_flat = tf.reshape(rnn_output, [-1, N])
target_flat = tf.reshape(target, [-1, m])
prediction_flat = tf.layers.dense(rnn_output_flat, m, activation=None)
prediction  = tf.reshape(prediction_flat, [-1, tf.shape(inputs)[1], m])

# Error function
loss = tf.reduce_sum(tf.square(target_flat - prediction_flat))
loss /= tf.cast(tf.reduce_sum(lengths), tf.float32)

# Optimizer
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Create graph and initialize variables
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.graph.finalize()  # Graph is read-only after this statement.

    # Do the learning
    alaki = np.zeros(epochs)
    for i in range(epochs):
        sess.run(train_step, feed_dict={
            inputs: train_X, target: train_Y, lengths: train_lengths})
        if i==0 or (i+1)%10==0:
            temp_loss = sess.run(loss,
                feed_dict={
                    inputs: train_X, target: train_Y, lengths: train_lengths})
            alaki[i]=temp_loss
            print(i+1, 'loss =', temp_loss)

    # Visualize modelling of training data
    model = sess.run(prediction, feed_dict={inputs: train_X, lengths: train_lengths})
    #plt.plot(train_X[0], label='train_x', color='lightgray', linestyle=':', linewidth=1)
    #plt.plot(train_Y[0], label='train_y', color='blue', linestyle=':', linewidth=1)
    #plt.legend(loc=1)
    #plt.xlabel('time [t]')
    #plt.ylabel('signal')
    #plt.xlim(-30, 250)
    #plt.show()
    # Visualize modelling of test data
    model = sess.run(prediction, feed_dict={inputs: test_X, lengths: test_lengths}) 
    model=model.astype('float')
    
    x_n=model[0,:].T
    x_n=x_n[0:500]
    ########################################################## majid_optimization
# Cost is a 1x1 matrix, we need a scalar.
    n = 500
    # Start with m and b initialized to 0s for the first try.
    #yhat= 1
    y0=np.ones(shape=(25,3))      
    cos=np.zeros(25)
    yha=y0[0,:]   
    for step in range(25):
        tmax=500
        t = np.arange(0.0, tmax, 1)
        R = odeint(two_lorenz_odes,yha , t)
        test, r2, r3 = R.T  # unpack columns
        test= inverse_utils.normalise(test) #normalise signal to range [-1, 1]

        diff = x_n - test
        dm = 5e-2 * (diff * x_n).sum() * 2 / n
        yha[0]= yha[0]-dm
        yha[1]= .1
        yha[2]= .1
        
        cost = np.mean(np.dot(diff.T, diff))
        print('cost =', cost)
        y0[step,0]=yha[0]
        y0[step,1]=yha[1]
        y0[step,2]=yha[2]      
        cos[step]=cost
    ind=np.where(cos == cos.min())
    ind=ind[0]
    maj=y0[ind,:]
    print('init =', maj)

    ###################################
    
    #plt.plot(test_X[0], label='input', color='lightgray', linestyle=':', linewidth=1)
    #plt.plot(test_Y[0], label='target_ref', color='blue', linestyle=':', linewidth=1)
    #plt.plot(model[0], label='prediction_ref',color='black',  linestyle=':',linewidth=1)
    #plt.legend(loc=1)
    #plt.xlabel('time [t]')
    #plt.ylabel('signal')
    #plt.xlim(-30, 250)
    #plt.show()

#########################3
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      Response
def two_lorenz_odes2(X, t):
    
    x1, y1, z1 = X
  
    dx1 = sigma * (y1 - x1)
    dy1 = -(x1 * z1) + (rho*x1) - y1
    dz1 = (x1 * y1 )- beta*z1

    return (dx1, dy1, dz1)


#y01 = [mov1, x02[0], x03[0]]
y01 = maj[0,:]
f = odeint(two_lorenz_odes2, y01, t)
x1, y1, z1 = f.T  # unpack columns
x1=inverse_utils.normalise(x1)
y1=inverse_utils.normalise(y1)
z1=inverse_utils.normalise(z1)
x=inverse_utils.normalise(x)
y=inverse_utils.normalise(y)
z=inverse_utils.normalise(z)

#plt.show()
#plt.plot(y1.T,label='Reconstructed yr(t)',color='black',  linestyle=':', linewidth=1)
#plt.legend(loc=1),
#plt.xlabel('t'),
#plt.ylabel('Reconstructed yr(t)')
#plt.xlim(-30, 250)
#plt.show()
#plt.plot(x.T -x1.T,  label='RNN-based Method Error & Sigma_learning=0.3',color='blue',linestyle=':', linewidth=1)
#plt.legend(loc=1),
#plt.xlabel('t'),
#plt.ylabel('Error e(t)')
#plt.xlim(-30, 250)


#plt.show()
#########################################################
a=alaki[9:800:10]

#plt.plot(range(9,800,10),a, 'o-', label='Temp LOSS',color='black',  linewidth=1)
#plt.legend(loc=1)
#plt.xlabel('epochs')
#plt.ylabel(' Loss ')
#plt.show()
#plt.savefig('rnn_convergence.pdf')

    
#plt.plot(train_X[0], label='Train Signal X', color='lightgray', linestyle=':', linewidth=1)
#plt.plot(train_Y[0], label= 'Target', color='blue', linestyle=':', linewidth=1)
#plt.legend(loc=2)
#plt.xlabel('time [t]')
#plt.ylabel('Signal')
#plt.xlim(0, 250)

plt.subplot(2, 1, 1)  
plt.plot(test_Y[0],label='Target', color='blue',  linewidth=1)
plt.plot(x1, '-', label='Denoised $\chi$ with RNN, $\sigma_n^2$=0.5',color='black',  linestyle=':',linewidth=1)
plt.legend(loc=2)
plt.xlabel('time ')
plt.ylabel('Signal')
plt.xlim(0, 250)


plt.subplot(2, 1, 2)  
plt.plot(x1.T, '-', label='Reconstructed $X_r$(t)',color='black',   linewidth=1)
plt.legend(loc=2),
plt.xlabel('time [t]'),
plt.ylabel('Reconstructed $X_r$')
plt.xlim(0, 250)
plt.savefig('rnn_den.pdf')
plt.show()

plt.plot(x[0:500].T-x1.T,'-', label='Error e(t) of RNN-based Method (X-$\hat X$) ',color='red', linewidth=.5)
plt.legend(loc=2),
plt.xlabel('time [t]'),
plt.ylabel('Error e(t)')
plt.xlim(0, 250)

plt.show()

 
plt.plot(z[0:500].T-z1.T,'-', label='Error e(t) of RNN-based Method (z-$\hat z$) ',color='red', linewidth=.5)
plt.legend(loc=2),
plt.xlabel('time [t]'),
plt.ylabel('Error e(t)')
plt.xlim(0, 250)

plt.show()

miangin=np.mean(abs(x[0:500]-x1))