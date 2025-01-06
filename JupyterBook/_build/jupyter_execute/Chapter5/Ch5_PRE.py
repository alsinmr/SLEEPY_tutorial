#!/usr/bin/env python
# coding: utf-8

# # <font  color = "#0093AF">Paramagnetic Relaxation Enhancement</font>

# <a href="https://githubtocolab.com/alsinmr/SLEEPY_tutorial/blob/main/ColabNotebooks/Chapter5/Ch5_PRE.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

# Stochastic motion can modulate the Hamiltonian, leading to magnetization decay towards thermal equilibrium. Electron relaxation is similarly a stochastic process acting on the nucleus via the hyperfine coupling, and therefore will also lead to nuclear relaxation. This process is referred to as Paramagnetic Relaxation Enhancement (PRE), and manifests in a number of forms. We will investigate the impact of electron $T_1$ and $T_2$ the nuclear $T_1$ and $T_2$ in the presence of an isotropic hyperfine coupling and a dipolar hyperfine coupling.
# 
# Note that modulation of the hyperfine coupling itself also leads to forms of paramagnetic relaxation, including the [Overhauser Effect](../Chapter4/Ch4_OverhauserEffect.ipynb), which was already discussed in the DNP chapter. We will not revisit these here, but be aware that hyperfine modulation will also play a role in transverse relaxation of the nucleus. 

# ## Setup

# In[143]:


import SLEEPY as sl
import numpy as np
import matplotlib.pyplot as plt
sl.Defaults['verbose']=False


# ## PRE from an isotropic hyperfine coupling
# 

# ### Build the system

# In[239]:


ex=sl.ExpSys(v0H=500,Nucs=['13C','e-'],LF=True)
aiso=5e5
ex.set_inter(Type='hyperfine',i0=0,i1=1,Axx=aiso,Ayy=aiso,Azz=aiso)    #Hyperfine coupling

L=ex.Liouvillian()


# ### Electron $T_2$ relaxation only

# In[240]:


L.clear_relax()
L.add_relax('T2',i=1,T2=1e-13)
seq=L.Sequence(Dt=1.01e-3)

rho=sl.Rho('13Cx+13Cz',['13Cp','13Cz','ez'])
_=rho.DetProp(seq,n=10000)


# In[241]:


rho.downmix()
T1=2.1
ax=plt.subplots(1,2,figsize=[9,4])[1]
rho.plot(axis='s',det_num=0,ax=ax[0])
ax[0].set_title('$T_2$ decay')
ax[0].plot(rho.t_axis,0.5*np.exp(-rho.t_axis/(T1*2)),color='black',linestyle=':')
rho.plot(axis='s',det_num=[1,2],ax=ax[1])
ax[1].set_title('Overhauser effect')
_=ax[1].plot(rho.t_axis,np.exp(-rho.t_axis/(T1/2))*0.5+0.5,color='black',linestyle=':')


# We see that the elecron $T_2$ induces transverse relaxation on the electron. In fact, this is an Overhauser effect in the rotating frame (ROE), but the short electron $T_2$ immediately destroys any magnetization gained on the nucleus. It also induces a transfer of longitudinal magnetization between electron and nucleus. In a real system, the electron $T_1$ relaxation would almost immediately destroy the gains, although in principle an Overhauser effect enhancement could be possible through this mechanism (although unlikely– the short electron $T_2$ used here would make electron saturation very difficult).
# 
# We have plotted a monoexponential curve on each plot. For $T_1$-decay, we have a time-constant of 2.1 seconds, and for $T_2$-decay, a time-constant twice as long, of 4.2 seconds. Then, this is a rare case when $T_2$ relaxation is slower than $T_1$ relaxation (see [Traficante 1991](https://doi.org/10.1002/cmr.1820030305)). This occurs when the relaxing field, in this case the non-secular components of the hyperfine coupling, only comes in the xy-plane, where the electron $T_2$ is active. We will later see that relaxation induced by the electron $T_1$ will prevent $T_2$ from actually exceeding $T_1$ in this example.
# 
# If we add an electron $T_1$, we can evaluate the dependence of the nuclear $T_1$ on the electron $T_2$. We expect to find the maximum $T_1$ relaxation when $T_{2e}=1/|(\omega_{0e}-\omega_{0n})|$, i.e. matched to the energy differnce of the two spins, which we will mark on the resulting plot.

# In[199]:


rho=sl.Rho('13Cz','13Cz')
T20=np.logspace(-14,-10,100)
R1=[]
for T2 in T20:
    L.clear_relax()
    L.add_relax('T1',i=1,T1=1e-6) #Longer than all T2s used
    L.add_relax('T2',i=1,T2=T2)
    R1.append(rho.extract_decay_rates(L.U(Dt=1e-3)))


# In[200]:


ax=plt.subplots()[1]
ax.loglog(T20,R1)
ax.set_ylim(ax.get_ylim())
T2max=1/(2*np.pi*np.abs(ex.v0[0]-ex.v0[1]))
ax.plot(T2max*np.ones(2),ax.get_ylim(),linestyle=':',color='black')
ax.set_xlabel(r'$T_{2e}$ / s')
_=ax.set_ylabel(r'$1/T_{1n}$ / s$^{-1}$')


# Indeed, the maximum occurs as expected.

# ### Electron $T_1$ relaxation only

# We next observe relaxation when only electron $T_1$ is present. This is unphysical, but we do it anyway to separate the two effects. Note that SLEEPY will warn us about the unphysicality.

# In[201]:


L.clear_relax()
L.add_relax('T1',i=1,T1=1e-13)
seq=L.Sequence(Dt=1.0e-3)

rho=sl.Rho('13Cx+13Cz',['13Cp','13Cz','ez'])
_=rho.DetProp(seq,n=10000)


# In[202]:


rho.downmix()
T1=2.1
ax=plt.subplots(1,2,figsize=[9,4])[1]
rho.plot(axis='s',det_num=0,ax=ax[0])
ax[0].set_title('$T_2$ decay')
ax[0].plot(rho.t_axis,0.5*np.exp(-rho.t_axis/(T1*2)),color='black',linestyle=':')
rho.plot(axis='s',det_num=[1,2],ax=ax[1])
_=ax[1].set_title('Overhauser effect')


# Then, we see that an electron $T_1$ acting via an isotropic hyperfine coupling induces $T_2$ decay, but no $T_1$ or Overhauser effect. As mentioned above, if we have electron $T_1$ and $T_2$ together, then we no longer have the case that $T_{2n}<T_{1n}.$ In fact, when $T_{1e}$=$T_{2e}$, then $T_{1n}$=$T_{2n}$. If $T_{1e}$ gets longer, then the $T_{2n}$ becomes shorter, giving the more typical case of $T_{1n}>T_{2n}$.
# 
# Below, we simulate the case that $T_{1e}=T_{2e}$, verifying that then $T_{1n}=T_{2n}$.

# In[230]:


L.clear_relax()
L.add_relax('T1',i=1,T1=1e-13)
L.add_relax('T2',i=1,T2=1e-13)
seq=L.Sequence(Dt=1.01e-3)

rho=sl.Rho('13Cx+13Cz',['13Cp','13Cz','ez'])
_=rho.DetProp(seq,n=10000)
rho.downmix()
T1=2.1
ax=plt.subplots(1,2,figsize=[9,4])[1]
rho.plot(axis='s',det_num=0,ax=ax[0])
ax[0].set_title('$T_2$ decay')
ax[0].plot(rho.t_axis,0.5*np.exp(-rho.t_axis/T1),color='black',linestyle=':')
rho.plot(axis='s',det_num=[1,2],ax=ax[1])
ax[1].plot(rho.t_axis,np.exp(-rho.t_axis/T1),color='black',linestyle=':')
_=ax[1].set_title('Overhauser effect')


# We can also acquire dependence of the nuclear $T_2$ on the electron $T_1$.
# 
# This is a little tricker than extracting the $T_1$. Decay rates are extracted from the real part of the eigenvalues of the propagators, but the relevant rates are easily identified because they are non-oscillating. For $T_2$, that is no longer the case, so that we must simply find the biggest term and its decay rate. This is not always the correct approach for extracting  decay of an oscillating signal, but works here.

# In[226]:


rho=sl.Rho('13Cx','13Cp')
T10=np.logspace(-12,-7,100)
R2=[]
for T1 in T10:
    L.clear_relax()
    L.add_relax('T1',i=1,T1=T1) #Longer than all T2s used
    L.add_relax('T2',i=1,T2=1e-12)
    rate,_,A=rho.extract_decay_rates(L.U(Dt=1e-3),mode='all')
    i=np.argmax(A[0])
    R2.append(rate[0][i])


# In[227]:


ax=plt.subplots()[1]
ax.loglog(T20,R2)
ax.set_ylim(ax.get_ylim())
ax.set_xlabel(r'$T_{1e}$ / s')
_=ax.set_ylabel(r'$1/T_{2n}$ / s$^{-1}$')


# Then, as the electron $T_1$ gets longer, the nuclear $T_2$ gets shorter, resulting in a linear plot.

# ## PRE from a dipolar hyperfine coupling

# A dipolar hyperfine coupling will also induce paramagnetic relaxation enhancement, although the dependence of the nuclear relaxation on the electron $T_2$ and $T_1$ is different than the scalar case. We investigate this below. Note that now we use a powder-averaged system and magic angle spinning.

# ### Build the system

# In[333]:


ex=sl.ExpSys(v0H=500,Nucs=['13C','e-'],LF=True,pwdavg=4)
delta=5e5
ex.set_inter(Type='hyperfine',i0=0,i1=1,Axx=-delta/2,Ayy=-delta/2,Azz=delta)    #Hyperfine coupling

L=ex.Liouvillian()


# ### Electron $T_2$ relaxation only

# In[344]:


L.clear_relax()
L.add_relax('T2',i=1,T2=5e-13)
seq=L.Sequence()
U=seq**20

rho=sl.Rho('13Cx+13Cz',['13Cp','13Cz','ez'])
_=rho.DetProp(U,n=10000)


# In[345]:


rho.downmix()
ax=plt.subplots(1,2,figsize=[9,4])[1]
rho.plot(axis='s',det_num=0,ax=ax[0])
ax[0].set_title('$T_2$ decay')
rho.plot(axis='s',det_num=[1,2],ax=ax[1])
_=ax[1].set_title(r'$T_1$, Overhauser effect')


# Electron $T_2$ then similarly induces a nuclear $T_2$ and $T_1$. Unlike the scalar case, the $T_1$ is not a pure Overhauser effect transfer, and eventually the full magnetization is destroyed. 

# The electron hyperfine coupling is strong enough to tilt the nucleus quantization away from the *z*-axis, such that MAS can no longer average the coupling, yielding a complex lineshape if we Fourier transform the transverse magnetization, shown below. Note that this effect will only appear if the nucleus is in the lab frame.

# In[346]:


rho.apod_pars['LB']=2
_=rho.plot(FT=True,det_num=0,apodize=True).set_xlim([200,-200])


# We can also sweep the electron $T_2$ to determine the dependence of the nuclear $T_1$ on the $T_2$.

# In[358]:


# Takes a few minutes
ex=sl.ExpSys(v0H=500,Nucs=['13C','e-'],LF=True,pwdavg=2)
delta=5e5
ex.set_inter(Type='hyperfine',i0=0,i1=1,Axx=-delta/2,Ayy=-delta/2,Azz=delta)    #Hyperfine coupling

L=ex.Liouvillian()

rho=sl.Rho('13Cz','13Cz')
T20=np.logspace(-14,-10,50)
R1=[]
for T2 in T20:
    L.clear_relax()
    L.add_relax('T2',i=1,T2=T2)
    R1.append(rho.extract_decay_rates(L.U()))


# In[359]:


ax=plt.subplots()[1]
ax.loglog(T20,R1)
ax.set_ylim(ax.get_ylim())
T2max=1/(2*np.pi*np.abs(ex.v0[0]-ex.v0[1]))
ax.plot(T2max*np.ones(2),ax.get_ylim(),linestyle=':',color='black')
ax.set_xlabel(r'$T_{2e}$ / s')
_=ax.set_ylabel(r'$1/T_{1n}$ / s$^{-1}$')


# As with the scalar hyperfine, we find the maximum at $T_{2e}=1/|(\omega_{0e}-\omega_{0n})|$.

# ### Electron $T_1$ relaxation only

# In[355]:


ex=sl.ExpSys(v0H=500,Nucs=['13C','e-'],LF=True,pwdavg=4)
delta=5e5
ex.set_inter(Type='hyperfine',i0=0,i1=1,Axx=-delta/2,Ayy=-delta/2,Azz=delta)    #Hyperfine coupling

L=ex.Liouvillian()


# In[356]:


L.clear_relax()
L.add_relax('T1',i=1,T1=5e-13)
seq=L.Sequence()
U=seq**20

rho=sl.Rho('13Cx+13Cz',['13Cp','13Cz','ez'])
_=rho.DetProp(U,n=10000)


# In[357]:


rho.downmix()
ax=plt.subplots(1,2,figsize=[9,4])[1]
rho.plot(axis='s',det_num=0,ax=ax[0])
ax[0].set_title('$T_2$ decay')
rho.plot(axis='s',det_num=[1,2],ax=ax[1])
_=ax[1].set_title(r'$T_1$, Overhauser effect')


# As with the scalar hyperfine coupling, the electron $T_1$ relaxation induces $T_2$ and $T_1$ relaxation on the nucleus without inducing any Overhauser effect. We can calculate the nuclear $T_1$ as a function of electron $T_1$

# In[366]:


# Takes a few minutes
ex=sl.ExpSys(v0H=500,Nucs=['13C','e-'],LF=True,pwdavg=2)
delta=5e5
ex.set_inter(Type='hyperfine',i0=0,i1=1,Axx=-delta/2,Ayy=-delta/2,Azz=delta)    #Hyperfine coupling

L=ex.Liouvillian()

rho=sl.Rho('13Cz','13Cz')
T10=np.logspace(-11,-7,50)
R1=[]
for T1 in T10:
    L.clear_relax()
    L.add_relax('T1',i=1,T1=T1)
    R1.append(rho.extract_decay_rates(L.U()))


# In[367]:


ax=plt.subplots()[1]
ax.loglog(T10,R1)
ax.set_ylim(ax.get_ylim())
T2max=1/(2*np.pi*np.abs(ex.v0[0]))
ax.plot(T2max*np.ones(2),ax.get_ylim(),linestyle=':',color='black')
ax.set_xlabel(r'$T_{2e}$ / s')
_=ax.set_ylabel(r'$(1/T_{1n})$ / s$^{-1}$')


# We see that the maximum occurs when $T_{1e}=1/|(\omega_{0n})|$, in contrast to the results from the scalar hyperfine coupling.

# In[ ]:




