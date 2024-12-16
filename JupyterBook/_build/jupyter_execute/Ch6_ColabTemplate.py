#!/usr/bin/env python
# coding: utf-8

# # <font  color = "maroon"> SLEEPY Colab Template
# Here we provide the basic SLEEPY setup in Google Colab, along with some suggestions on getting started.

# <a href="https://githubtocolab.com/alsinmr/SLEEPY_tutorial/ColabTemplate.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

# ## Setup

# In[3]:


# SETUP SLEEPY
get_ipython().system('git clone https://alsinmr.github.io/SLEEPY')


# In[4]:


import SLEEPY as sl
import numpy as np
import matplotlib.pyplot as plt


# ## Run the simulation

# In[27]:


# Experimental system settings (use two or more for exchange)
ex0=sl.ExpSys(v0H=...,Nucs=[...,...],vr=60000)
ex0.set_inter('...',i0=...,i1=..,...)
ex1=ex0.copy()

# Build the Liouvillian
L=sl.Liouvillian(ex0,ex1,kex=sl.Tools.twoSite_kex(...))

# Add a sequence
seq=L.Sequence()
# Add a channel
seq.add_channel('...',t=...,v1=...,phase=...,voff=...)

# Density matrix
rho=sl.Rho(rho0='...',detect='...')

# Run the simulation
rho.DetProp(seq,n=...,n_per_seq=...)

# Plot the results
rho.plot(FT=...)

