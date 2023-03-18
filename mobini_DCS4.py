
import tensorflow as tf
from scipy.integrate import odeint
from ipywidgets import interactive
from IPython.display import display
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import Gen_sonar as s
import torch
import inverse_utils
import dip_utils
import time
import pywt
import scipy.fftpack as spfft
from scipy.interpolate import interp1d
N=1
angle=0.0
tmax=102.4
sigma=10.0
beta=8./3
rho=28.0
t = np.arange(0.0, tmax, .1)

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



#####################################################

#NETWORK SETUP
LR = 1e-4 # learning rate
MOM = .9 # momentum
NUM_ITER = 800 # number iterations
WD = 1 # weight decay for l2-regularization
Z_NUM = 32 # input seed
NGF = 64 # number of filters per layer
nc = 1 #num channels in the net I/0

alpha_tv = 1e-1 #TV parameter for net loss
LENGTH = 1024


#CUDA SETUP
CUDA = torch.cuda.is_available()
print("On GPU: ", CUDA)

if CUDA :
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


#SIGNAL GENERATION
org =x[0:200].T
orgy=inverse_utils.normalise(org)
orgyz=inverse_utils.normalise(z.T)

#signal += np.random.normal(loc = 0, scale = .5, size=LENGTH) #add background noise
np.random.seed(0)
signal=org+np.random.normal(0, .5,200)
signal = inverse_utils.normalise(signal) #normalise signal to range [-1, 1]


m = np.zeros((LENGTH, 1))
m[0:200, 0] = signal

plt.figure()
plt.plot(range(200), signal)
plt.title("Noisy Signal")
plt.show()

#IMPUTATION SETUP
missing1 = range(3200,3256) #usually (395, 425)
missing_samples = np.array(missing1)
kept_samples = [m for m in range(LENGTH) if m not in missing_samples]
A = np.identity(LENGTH)[kept_samples, :] #the dropout transformation
y = m[kept_samples]

#DIP imputation
x_pat = dip_utils.run_DIP_short(A, y, dtype, NGF = NGF, LR=LR, MOM=MOM, WD=WD, output_size=LENGTH, num_measurements=len(kept_samples), CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv)

x_hat=x_pat.astype('float')
########################################################################
x_n=x_hat[0:200].T
 ########################################################## Genetic_optimization
import ga


# Inputs of the equation.
equation_inputs = [1]

# Number of the weights we are looking to optimize.
num_weights = len(equation_inputs)

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 10000
num_parents_mating = 100

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_
#per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
new_population = np.random.uniform(low=0, high=.2, size=pop_size)
print(new_population)

"""
new_population[0, :] = [2.4,  0.7, 8, -2,   5,   1.1]
new_population[1, :] = [-0.4, 2.7, 5, -1,   7,   0.1]
new_population[2, :] = [-1,   2,   2, -3,   2,   0.9]
new_population[3, :] = [4,    7,   12, 6.1, 1.4, -4]
new_population[4, :] = [3.1,  4,   0,  2.4, 4.8,  0]
new_population[5, :] = [-2,   3,   -7, 6,   3,    3]
"""
best_outputs = []
num_generations = 10
for generation in range(num_generations):
    print("Generation : ", generation)
    # Measuring the fitness of each chromosome in the population.
    fitness = ga.cal_pop_fitness(x_n,new_population)
    print("Fitness")
    print(fitness)

    
    
    # Selecting the best parents in the population for mating.
    parents = ga.select_mating_pool(new_population, fitness, 
                                      num_parents_mating)
    print("Parents")
    print(parents)

    # Generating next generation using crossover.
    offspring_crossover = ga.crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_weights))
    print("Crossover")
    print(offspring_crossover)

    # Adding some variations to the offspring using mutation.
    offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)
    print("Mutation")
    print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
    
# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = ga.cal_pop_fitness(x_n, new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])


import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()


    ####################################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      Response

def two_lorenz_odes2(X, t):
    
    x1, y1, z1 = X
  
    dx1 = sigma * (y1 - x1)
    dy1 = -(x1 * z1) + (rho*x1) - y1
    dz1 = (x1 * y1 )- beta*z1

    return (dx1, dy1, dz1)
maj1=new_population[best_match_idx, :][0,0,0]


#y01 = [mov1, x02[0], x03[0]]
y01 = [maj1, .1, .1]
f = odeint(two_lorenz_odes2, y01, t)
x1, y1, z1 = f.T  # unpack columns
x1=inverse_utils.normalise(x1)
y1=inverse_utils.normalise(y1)
z1=inverse_utils.normalise(z1)


############################################################################
#Plotting signals before imputation
plt.figure()
plt.plot( x_hat[0:200], label = "DCS", color = 'b', linestyle=':', linewidth=1)
plt.legend(loc=1),



#plt.title("Deep Image Prior")
plt.show()

###########################################################################

plt.subplot(2, 1, 1)
plt.plot(orgy, label = "Target", color = 'c', linestyle='-', linewidth=1)
plt.legend(loc=2),
plt.xlim(0, 200)

#plt.subplot(2, 1, 2)  
#plt.plot(range(LENGTH), signal, label = "Noisy Signal- $\sigma_n^2=0.5$", color = 'b', linestyle=':', linewidth=1)
#plt.legend(loc=2),
#plt.xlim(0, 250)
#plt.show()


plt.plot( x_hat[0:200], label = 'Denoised $\hat x$ with DCS , $\sigma_n^2$=0.5', color = 'k', linestyle=':', linewidth=1)
plt.legend(loc=2),
plt.xlim(0, 200)
plt.xlabel('time [t]'),
plt.ylabel('Signal')
plt.savefig('Den_dcs.pdf')
plt.show()


plt.plot((x_hat[0:200]-orgy),label='Error of DCS Method (x-$\hat x$)', color = 'r', linestyle=':', linewidth=1)
plt.legend(loc=2),
plt.xlabel('time [t]'),
plt.ylabel('Error e(t)')
plt.xlim(0, 200)
plt.ylim(-2, 2)
plt.show()
miangin=np.mean(abs(x_hat[0:200]-orgy))
#########################################################
#####################################################
 
iter=[200, 400, 600, 1000 ,1400, 1800, 2200, 2600, 3000]
error=[.042,.021 , .0144, .0149, .0142, .0145, .0149, .0146, .0148]

plt.subplot(2, 1, 1) 
plt.plot(iter,error,'o-', label='Average Error-DCS vs Number of Iter.', color = 'k', linestyle=':', linewidth=1)

plt.legend(loc=1),
plt.xlabel('Iteration')
plt.ylabel('Error e(t)')
#plt.xlim(0, 530)
#plt.ylim(-.5, .5)
plt.savefig('Av_err2.pdf')

#########################################################
#####################################################
sigma=[0,.1,.2,.3,.4,.5,.6,.7,.8]
dwist=[.015,.025, .026, .0307, .033, .036, .037, .041, .056 ]
hashsad=[.012,.011, .0146, .0142, .016, .0196, .0183, .0281, .033 ]
dozar=[.004512,.005411, .007, .01, .0126, .0149, .0178, .0276, .0288 ]
plt.subplot(2, 1, 2)  
plt.plot(sigma,dwist,'s-', label='DCS (#iter.=200)', color = 'b', linestyle=':', linewidth=2)
plt.plot(sigma,hashsad, 'o-',label='DCS (#iter.=800)', color = 'c', linestyle=':', linewidth=2)
plt.plot(sigma,dozar, '*-',label='DCS (#iter.=2000)', color = 'k', linestyle=':', linewidth=2)

plt.subplot(2, 1, 2) 
plt.legend(loc=2),
plt.xlabel('$\sigma_n^2$'),
plt.ylabel('Error e(t)')
#plt.xlim(0, 530)
#plt.ylim(-.5, .5)

plt.savefig('Av_err2.pdf')
plt.show()
################################################
##################################################
###################################################
plt.plot(orgyz,label='DCS- Drive Signal $z$, $\sigma_n^2=0.5$ ',color='black',  linestyle=':', linewidth=1)
plt.legend(loc=4),
plt.xlabel('t'),
plt.ylabel('Signal')
plt.xlim(0, 200)


plt.plot(z1,label='DCS- Response Signal $z_r$, $\sigma_n^2=0.5$',color='blue',  linestyle='-', linewidth=1)
plt.legend(loc=4),
#plt.xlabel('t'),
plt.ylabel('Signal')
plt.xlim(0, 200)
plt.savefig('DCS_Response.pdf')
plt.show()

plt.plot(y1[0:200]-orgy[0:200],label='Synchronization Error of DCS (Y-$\hat Y$)- $\sigma_n^2=0.5$',color='red',  linestyle='-', linewidth=1)
plt.legend(loc=4),
plt.xlabel('t'),
plt.ylabel('Error e(t)')
plt.xlim(0, 200)
plt.ylim(-2,2)
#%%%%%%%%%%%%
plt.show()



