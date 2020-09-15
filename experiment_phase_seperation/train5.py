#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import sympy
from sympy import lambdify
import numpy as np
import torch
torch.manual_seed(12)
torch.cuda.manual_seed(12)
torch.cuda.manual_seed_all
dtype = torch.float
device = torch.device("cuda:0")
import sys
sys.path.append('/u/wangnisn/devel/machine_learning_kinetics/ml_kinetics')
from mlk import learning

u, u_xx, u_yy = sympy.symbols('u u_xx u_yy')
order = 2
RK = []
for k in range(order+1):
    RK += [u*(1-u)*(2*u-1)**k,]
dictionary = (u_xx+u_yy, ) + tuple(RK)
u = np.load('u_all.npy')
t_mesh = np.arange(u.shape[0])
x_mesh = np.arange(u.shape[1])
y_mesh = np.arange(u.shape[2])
t,x,y = np.meshgrid(t_mesh, x_mesh, y_mesh, indexing='ij')
t = np.reshape(t,(-1,1))
x = np.reshape(x,(-1,1))
y = np.reshape(y,(-1,1))
u = np.reshape(u,(-1,1))

t_max = np.max(t)
x_max = np.max(x)
y_max = np.max(y)
t = t / t_max
#x = x/x_max
#y = y/y_max
#scalers = torch.tensor([1/x_max,  1/t_max],device=device,dtype=dtype)
scalers = None
inputs = {'x':torch.tensor(x,device=device,dtype=dtype,requires_grad=True),
          'y':torch.tensor(y,device=device,dtype=dtype,requires_grad=True),
          't':torch.tensor(t,device=device,dtype=dtype,requires_grad=True)
                              }
u  = torch.tensor(u,device=device,dtype=dtype)

params = { 'n_epochs':20000,
           'batch_size':500000,
           'alpha_pde_start':0.1,
           'alpha_pde_end':0.1,
           'alpha_l1':1e-7, 
           'scalers':scalers,
           'linearRegInterval':100,
           'linearRegression':False,
           'linearRegStart':490,
           'warmup_nsteps':1000,
           'width':30,
           'layers':6,
           'lr':0.001,
           'lr_multi_factor':0.9999,
           'fixed_coefs':{},
           'fit_intercept':False,
           'initializing_method':'zero',
           'activation':'tanh',
           'device':device,
           'log_batch':False,
           'logfile':'train5.txt',
           'save_best':True,
           'best_model_filename':'best_model_train5.pt',
           'best_coefs_filename':'best_coefs_train5.txt'}

model = learning(inputs=inputs, u=u, dictionary=dictionary,**params)

