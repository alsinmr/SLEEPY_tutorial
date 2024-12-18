#!/usr/bin/env python
# coding: utf-8

# # <font  color = "#0093AF"> Relaxation in complex sequences: RECRR

# <a href="https://githubtocolab.com/alsinmr/SLEEPY_tutorial/blob/main/ColabNotebooks/Chapter3/Ch3_RECRR.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

# A challenge in acquiring $R_{1\rho}$ relaxation rate constants is that coherent oscillation is present at the beginning of the $R_{1\rho}$ period, which distorts the signal decay. While we could wait until oscillation is subsided, we lose significant signal. The beginning of the relaxation period is also particularly important, because $R_{1\rho}$ relaxation is multiexponential. The initial slope of decay gives the averaged rate constant, which is what we would like to acquire. However, after time, the faster relaxing components have decayed more than the slower components, so that the slope no longer correctly represents the correct average.
# 
# Keeler et al. propose a solution to suppress oscillation at the beginning of the $R_{1\rho}$ period, referred to as the REfocused CSA Rotating-frame Relaxation experiment (RECRR). In this experiment, the spin-locks (CW$_{\pm x}$)  are switched in phase, and $\pi$-pulses are inserted to invert the magnetization, as follows:$^1$ 
# 
# CW$_x$ - $\pi_y$ - CW$_{-x}$ – CW$_{-x}$ - $\pi_{-y}$ CW$_x$
# 
# The spin-locks each have an integer number of rotor periods.
# 
# We will investigate the RECRR here and compare its performance to the standard $R_{1\rho}$ experiment
# 
# [1] E.G. Keeler, K.J. Fritzsching, A.E. McDermott. *[J. Magn. Reson](https://doi.org/10.1016/j.jmr.2018.09.004)*, **2018**, 296, 130-137.

# ## Setup

# In[ ]:


# SETUP SLEEPY


# In[102]:


import SLEEPY as sl
import numpy as np
import matplotlib.pyplot as plt


# ## Build the system and Liouvillian

# In[103]:


ex0=sl.ExpSys(v0H=600,Nucs=['15N','1H'],vr=16000,pwdavg=sl.PowderAvg(),n_gamma=50)
ex0.set_inter('dipole',i0=0,i1=1,delta=44000)
ex0.set_inter('CSA',i=0,delta=113,euler=[0,23*np.pi/180,0])
ex1=ex0.copy()
ex1.set_inter('dipole',i0=0,i1=1,delta=44000,euler=[0,15*np.pi/180,0])
ex1.set_inter('CSA',i=0,delta=113,euler=[[0,23*np.pi/180,0],[0,15*np.pi/180,0]])

L=sl.Liouvillian(ex0,ex1)
L.kex=sl.Tools.twoSite_kex(tc=200e-6)


# ## Build the propagators and density matrices.
# We build the spin-locks on $x$ and $-x$, and $\pi$-pulses on $y$ and $-y$
# 
# Density matrix for $R_{1\rho}$ and RECRR sequences are also constructed, including a basis set reduction for propagators and density matrices.

# In[104]:


Ux=L.Sequence().add_channel('15N',v1=25000).U()
Umx=L.Sequence().add_channel('15N',v1=25000,phase=np.pi).U()
Upiy=L.Udelta('15N',phase=np.pi/2)
Upimy=L.Udelta('15N',phase=3*np.pi/2)

R1p=sl.Rho('15Nx','15Nx')

R1p,Ux,Umx,Upiy,Upimy=R1p.ReducedSetup(Ux,Umx,Upiy,Upimy)
RECRR=R1p.copy_reduced()


# The basic $R_{1\rho}$ experiment only requires repetition of the `Ux` propagator, so we can simply use the `DetProp` function

# ## Propagation and plotting
# We start with the standard $R_{1\rho}$ experiment

# In[105]:


r1p,A=R1p.extract_decay_rates(Ux,mode='avg')
#sc is the fraction of the total signal that undergoes decay without oscillation
sc=(A*ex0.pwdavg.weight).sum()  
R1p.DetProp(Ux,n=800)
_=R1p.plot()


# At the beginning of the decay, we observe a loss of almost half of the signal, due to coherent contributions.
# 
# We also calculate the RECRR sequence. Since additional rotor periods must be inserted into each of four spin-lock blocks, we cannot use the `DetProp` function. We do accelerate the calculation by building up the $x$ and $-x$ spin locks at each loop step, instead of recalculating `Ux**k` and `Umx**k` at every step.

# In[106]:


Umx0=Umx
Ux0=Ux
RECRR()
for k in range(200):
    RECRR.reset()
    (Umx0*Upimy*Umx0*Ux0*Upiy*Ux0*RECRR)()
#     (Ux0*Upiy*Ux0*RECRR)()
    Umx0=Umx*Umx0
    Ux0=Ux*Ux0
_=RECRR.plot()


# Indeed, oscillations at the beginning of the decay curve have been almost entirely removed. However, it is worth noting that the decay rate at the beginning of the curve appears to be faster than with the standard $R_{1\rho}$ experiment. We overlay the two curves, scaling the RECRR curve to match the beginning of the non-coherent decay of the $R_{1\rho}$ curve.

# In[107]:


ax=R1p.plot()
ax.plot(RECRR.t_axis*1e3,RECRR.I[0].real*sc)
ax.legend((r'$R_{1\rho}$','RECRR'))


# We clearly see that the RECRR curve is relaxing more quickly at the beginning.
# 
# It can be seen more clearly what is going on if we consider a single crystal, where the $R_{1\rho}$ experiment should be closer to monoexponential.

# ## Single crystal behavior
# We simulate both sequences for a single crystal, sweeping the correlation time. At shorter correlation times, the relaxation behavior is very similar between the two sequences. However, at longer correlation times, a fast decaying component emerges in the RECRR that is not present in the standard $R_{1\rho}$ experiment. This is likely because the RECRR experiment relies on a refocusing of the CSA (and dipole) by cycling the phases, where if the frequency of the phase change matches the rate of motion, we obtain an additional relaxation mechanism that is not present in the $R_{1\rho}$ experiment. Since the frequency of the phase change is varying throughout the RECRR experiment, the timescale sensitivity is also varying.
# 
# We plot the $R_{1\rho}$ curves at the top, and RECRR curves at the bottom. We calculate the $R_{1\rho}$ decay without oscillation as well (dashed lines). This curve is also overlayed over the RECRR curve with a scaling factor. We do this to show that the relaxation at longer times is very similar to the $R_{1\rho}$ behavior. For slower motion, it takes longer for the two curves to line up, since we are further into the sequence before the interference between motion and phase changes vanishes.

# In[108]:


fig,ax0=plt.subplots(2,5,figsize=[12,5])
for tc,ax,sc in zip([1e-6,1e-5,1e-4,1e-3,1e-2],ax0.T,[1,1,.65,.55,.05]):
    ex0=sl.ExpSys(v0H=600,Nucs=['15N','1H'],vr=16000,pwdavg=sl.PowderAvg()[10],n_gamma=50)
    ex0.set_inter('dipole',i0=0,i1=1,delta=44000)
    ex0.set_inter('CSA',i=0,delta=113,euler=[0,23*np.pi/180,90*np.pi/180])
    ex1=ex0.copy()
    ex1.set_inter('dipole',i0=0,i1=1,delta=44000,euler=[0,15*np.pi/180,0])
    ex1.set_inter('CSA',i=0,delta=113,euler=[[0,23*np.pi/180,90*np.pi/180],[0,15*np.pi/180,0]])

    L=sl.Liouvillian(ex0,ex1)
    L.kex=sl.Tools.twoSite_kex(tc=tc)

    Ux=L.Sequence().add_channel('15N',v1=25000).U()
    Umx=L.Sequence().add_channel('15N',v1=25000,phase=np.pi).U()
    Upiy=L.Udelta('15N',phase=np.pi/2)
    Upimy=L.Udelta('15N',phase=3*np.pi/2)

    R1p=sl.Rho('15Nx','15Nx')

    R1p,Ux,Umx,Upiy,Upimy=R1p.ReducedSetup(Ux,Umx,Upiy,Upimy)
    RECRR=R1p.copy_reduced()

    A,r1p=R1p.extract_decay_rates(Ux,mode='rates')
    A,r1p=A[0],r1p[0]
    R1p.DetProp(Ux,n=800)

    R1p.plot(ax=ax[0])

    # Here we calculate the oscillation-free decay from R1p
    I=np.sum([A0*np.exp(-r1p0*R1p.t_axis) for A0,r1p0 in zip(r1p,A)],axis=0)

    ax[0].plot(R1p.t_axis*1e3,I,linestyle=':',color='black')
    ax[0].set_title(fr'$\tau_c$ = {tc*1e6:.0f} $\mu$s')
    

    Umx0=Umx
    Ux0=Ux
    RECRR()
    for k in range(200):
        RECRR.reset()
        (Umx0*Upimy*Umx0*Ux0*Upiy*Ux0*RECRR)()
        Umx0=Umx*Umx0
        Ux0=Ux*Ux0
    RECRR.plot(ax=ax[1])
    ax[1].plot(R1p.t_axis*1e3,I/I[0]*sc,linestyle=':',color='black')
    for a in ax:a.set_ylim([0,1])
fig.tight_layout()
_=ax0[0,0].legend((r'$R_{1\rho}$',r'Oscillation Removed'))


# Then, while the RECRR pulse sequence presents some significant advantages over the $R_{1\rho}$ sequence, because of the more complex relaxation behavior, it becomes necessary to use simulation to fit the curves, rather than relying on simple mono-exponential or bi-exponential fitting. This is especially important at slower correlation times, where a second relaxation mechanism emerges, and the sensitivity of this experiment to correlation time varies throughout the duration of the experiment.

# In[ ]:




