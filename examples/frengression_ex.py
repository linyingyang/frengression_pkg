# NOTES
# use 0.1.13 engression?
# add continuous covariates instead of binary if needed--binary is harder for package?
# noise_dim annot be larger than number of covariates
# hidden_dim = more means longer run time
# learning rate= if larger, trains faster but may not converge


import torch
import pandas as pd
import sys, os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import copy
import warnings
import rpy2
from utils import *
from frengression import *
device = torch.device('cpu')


df = pd.read_csv('./data/fakedataset_simcausal_complex.csv')

print(df.head())
df.info()

# Dummy data
s = df[['L1_0']].to_numpy(dtype=float)
x = df[['A1_0','A1_1','A1_2']].to_numpy(dtype=float)
y = df[['Y_1','Y_2','Y_3']].to_numpy(dtype=float)
z = df[['L2_0','L2_1','L2_2']].to_numpy(dtype=float)

s_tr = torch.tensor(s, dtype=torch.float32)
x_tr = torch.tensor(x, dtype=torch.int32)
y_tr = torch.tensor(y, dtype=torch.float32)
z_tr = torch.tensor(z, dtype=torch.float32)


model = FrengressionSeq(x_dim=1, y_dim=1, z_dim=1, T=3, s_dim = 1, noise_dim=1, 
                        num_layer=300, hidden_dim=50,#100 
                        device=device, x_binary = True, s_in_predict=True)

model.train_e(s=s_tr, x=x_tr,z=z_tr,num_iters=100, lr=1e-4, print_every_iter=1000)

y_margin_sample=model.sample_causal_margin(s=torch.tensor([[0]],dtype=torch.float32), 
                                            x = torch.tensor([[1]*5],dtype=torch.int32),
                                            sample_size=1000)
y_margin_sample.head()
type(y_margin_sample)

def five_point_summary(arr):
  arr = np.asarray(arr)
  return {
  "min": np.nanmin(arr),
  "q1": np.nanpercentile(arr, 25),
  "median": np.nanpercentile(arr, 50),
  "q3": np.nanpercentile(arr, 75),
  "max": np.nanmax(arr)
  }
five_point_summary(y_margin_sample[0])
y_tr[0]
five_point_summary(y_margin_sample[1])
five_point_summary(y_margin_sample[2])
len(y_margin_sample[2])
sampled_z_5x = FrengressionSeq.sample_joint(self=model,s=torch.tensor([[0]],dtype=torch.float32),sample_size=1000)
model.sample_joint(s=torch.tensor([[0]],dtype=torch.float32),sample_size=100)
sampled_z_5x.head()
# sampled_x, sampled_y, sampled_z = model.sample_joint(sample_size = int(1000))
