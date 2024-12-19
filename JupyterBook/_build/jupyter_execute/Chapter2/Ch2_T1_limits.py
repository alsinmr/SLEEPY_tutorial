#!/usr/bin/env python
# coding: utf-8

# # <font  color = "#0093AF"> $T_1$ limits

# <a href="https://githubtocolab.com/alsinmr/SLEEPY_tutorial/blob/main/ColabNotebooks/Chapter2/Ch2_T1_limits.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

# Longitudinal relaxation occurs because a relative fast motion—on the timescale of the lab frame rotation—induces a slow evolution of the density matrix. In this notebook, we check the validity of this lab frame calculation against analytical formulas, and also investigate the range in which equilibrium of the system to thermal equilibrium (via "DynamicThermal") is valid.

# ## SETUP

# In[ ]:


# SETUP SLEEPY


# In[2]:


import SLEEPY as sl
import numpy as np
import matplotlib.pyplot as plt
# !git clone https://github.com/alsinmr/pyDR     #Uncomment on Google Colab to import pyDR. 
#pyDR will also install MDAnalysis
import pyDR


# ## Build the system

# We mimick a tumbling motion by hopping around the 'rep10' power average. Note that tumbling is currently only implemented for colinear tensors without asymmetry (we don't include a gamma average, so this yield vector tumbling, not tensor tumbling).

# In[3]:


# Since we use a tumbling model, we only need a single angle in the powder average
ex0=sl.ExpSys(v0H=400,Nucs=['15N','1H'],vr=0,LF=True,pwdavg='alpha0beta0')
delta=sl.Tools.dipole_coupling(.102,'1H','15N')
ex0.set_inter('dipole',i0=0,i1=1,delta=22954.8)

# Set up tumbling
q=2
ex,kex=sl.Tools.SetupTumbling(ex0,q=q,tc=1e-9)  #This hops around the rep10 powder average

L=sl.Liouvillian(*ex,kex=kex)
# L.add_relax('DynamicThermal')
seq=L.Sequence(Dt=.1)


# We plot the full Liouvillian below, just to give an idea what the exchange looks like.

# In[4]:


ax=L.plot()
ax.figure.set_size_inches([8,8])


# ## Sweep the correlation time
# We don't use explicit propagation. Instead, we extract the decay rates using `rho.extract_decay_rates` to find the $T_1$ decay. We also calculate the signal at equilibrium. This is done by raising the density matrix to an infinite power (internally, we don't really use infinity- this just triggers an algorithm to calculate the equilibrium position)

# In[4]:


tc0=np.logspace(-5,-13,80)
rho=sl.Rho('15Nz','15Nz')
R1=[]
for tc in tc0:
    rho.reset()
    L.kex=sl.Tools.SetupTumbling(ex0,q=q,tc=tc)[1]
    R1.append(rho.extract_decay_rates(seq))


# ## Compare to analytical formula

# In[10]:


ax=plt.subplots()[1]
ax.semilogx(tc0,R1,color='red')
nmr=pyDR.Sens.NMR(v0=400,Nuc='15N',CSA=0,Type='R1')
ax.semilogx(nmr.tc,nmr.rhoz.T,color='black',linestyle=':',linewidth=3)
ax.set_xlabel(r'\tau_c / s')
ax.set_ylabel(r'R_1 / s')
ax.legend(('Simulation','Analytic'))


# We see extremely good agreement between the simulation and the analytical formulas implemented in pyDR

# ## Test "DynamicThermal" performance

# SLEEPY contains a function to bring a system relaxed by exchange back into thermal equilibrium, obtained by running `L.add_relax('DynamicThermal')`. This is a somewhat artificial correction to obtain thermal equilibrium, and has a more limited application; this is because we have essentially very fast dynamics applied via `L.kex`, but then a correction that slowly brings magnetization back into the system. Small numerical errors can lead to large variation in the resulting equilibrium values.
# 
# We test this here as a function of correlation time. We do not recommend using 'DynamicThermal' with a large number of states in exchange.

# In[3]:


ex0=sl.ExpSys(v0H=400,Nucs=['15N','1H'],vr=0,LF=True,pwdavg='alpha0beta0')
delta=sl.Tools.dipole_coupling(.102,'1H','15N')
ex0.set_inter('dipole',i0=0,i1=1,delta=22954.8)
ex1=ex0.copy().set_inter('dipole',i0=0,i1=1,delta=22954.8,euler=[0,np.pi/8,0])

L=sl.Liouvillian(ex0,ex1,kex=sl.Tools.twoSite_kex(1e-8))
L.add_relax('DynamicThermal')

seq=L.Sequence(Dt=.1)

rho=sl.Rho('Thermal',['15Nz','1Hz'])


# In[12]:


tc0=np.logspace(-7,-11,80)
Ieq=[]
R1=[]
for tc in tc0:
    rho.clear()
    L.kex=sl.Tools.twoSite_kex(tc=tc)
    U=seq.U()
    R1.append(rho.extract_decay_rates(U))
    #     U=seq.U()
    Ieq.append((U**np.inf*rho)().I[:,0].real)

Ieq=np.array(Ieq)


# In[13]:


ax=plt.subplots(1,2,figsize=[9,5])[1]
ax[0].semilogx(tc0,Ieq)
ax[0].semilogx([tc0[0],tc0[-1]],np.ones(2)*ex0.Peq[0],color='black',linestyle=':')
ax[0].semilogx([tc0[0],tc0[-1]],np.ones(2)*ex0.Peq[1],color='grey',linestyle='--')
ax[0].set_ylim([ex0.Peq[0]*15,ex0.Peq[1]*5])
ax[1].semilogx(tc0,R1)


# In[69]:


L.kex=sl.Tools.twoSite_kex(tc=5e-11)
# L.add_relax('DynamicThermal')
U=seq.U()
U=L.U(Dt=100)
U.calcU()
rho=sl.Rho('zero',['15Nz','1Hz'])
for _ in range(2):
    (rho().prop(U))()
# rho.DetProp(U,n=1000)
ax=rho.plot(axis='s')
ax.plot([0,rho.t_axis[-1]],np.ones(2)*ex0.Peq[0],color='black',linestyle=':')
ax.plot([0,rho.t_axis[-1]],np.ones(2)*ex0.Peq[1],color='grey',linestyle='--')


# In[72]:


U=L.U(Dt=100)
rho=sl.Rho('Thermal',['1Hz','15Nz'])
rho.L=L
np.abs((U[0]@U[0]@U[0]@rho._rho0[0])-rho._rho0[0]).max()
# (U*U*U*rho)().I


# In[64]:


rho.


# In[ ]:




