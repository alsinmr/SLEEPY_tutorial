#!/usr/bin/env python
# coding: utf-8

# # <font  color = "#0093AF"> Exchange Spectroscopy (EXSY)

# In the previous example, we simulated exchange in a 1D spectrum. Here, we perform the 2D EXSY experiment, under exchange conditions. We will then look how it results in a 2D EXSY spectrum, and how that spectrum changes as a function of a mixing time.

# ## Setup

# In[1]:


import os
os.chdir('../../../')
import SLEEPY as sl
import numpy as np
import matplotlib.pyplot as plt


# ## Build the system
# The first step is to build the system, which will have a single nucleus, with two different chemical shifts.

# In[2]:


ex0=sl.ExpSys(Nucs='13C',v0H=600)    #We need a description of the experiment for both states (ex0, ex1)
ex1=ex0.copy()
ex0.set_inter(Type='CS',i=0,ppm=-5)   #Add the chemical shifts
_=ex1.set_inter(Type='CS',i=0,ppm=5)


# ## Add the exchange process
# First, export this sytem into Liouville space, allowing us to introduce an exchange process. Then we'll define a correlation time and population 1 and population 2. From this we can build the exchange matrix and append it to the Liouvillian. We also add some $T_2$ relaxation to destroy transverse magnetization during the delay period for exchange and produce some broadening.

# In[3]:


L=sl.Liouvillian((ex0,ex1))           #Builds the two different Hamiltonians and exports them to Liouville space

tc=1     #Correlation time (10 s)
p1=0.75  #Population of state 1
p2=1-p1  #Population of state 2

L.kex=sl.Tools.twoSite_kex(tc=tc,p1=p1)

_=L.add_relax(Type='T2',i=0,T2=.01)


# We can check the correlation time and equilibrium populations based on eigenvalue decomposition of the exchange matrix

# In[4]:


d,v=np.linalg.eig(L.kex) #d is eigenvalues, v is eigenvectors
i=np.argsort(d)  # One element should be zero, the other negative
tc_k=-1/d[i[0]]
peq=v[:,i[1]]
peq/=peq.sum()
print(f'The correlation time is {tc_k:.1f} s, and the equilibrium populations are ({peq[0]:.2f},{peq[1]:.2f})')


# ## Run as a 2D experiment
# First, we'll just calculate one 2D spectrum, and then later check how the spectrum evolves as a function of a delay time. We need an initial density matrix, $S_x$, a detection matrix, $S^+$, and propagators for evolution times, $\pi$/2 pulses, and a delay for the exchange process. We start with generating the propagators and density matrices.

# In[5]:


rho=sl.Rho(rho0='S0x',detect='S0p')

#Pulse sequences for pi/2 pulses
v1=50000
tpi2=1/v1/4
pi2x=L.Sequence() 
pi2y=L.Sequence()
pi2my=L.Sequence()
pi2x.add_channel('13C',t=[0,tpi2],v1=v1,phase=0)      #Here we add the pi/2 pulse on x
pi2y.add_channel('13C',t=[0,tpi2],v1=v1,phase=np.pi/2) #Here we add the pi/2 pulse on y
pi2my.add_channel('13C',t=[0,tpi2],v1=v1,phase=-np.pi/2) #Here we add the pi/2 pulse on -y

Upi2x=pi2x.U()  #Get the propagators for the pulse sequence
Upi2y=pi2y.U()
Upi2my=pi2my.U()

Dt=1/(2*10*150)  #For a 10 ppm shift difference, this should be enough to easily capture both peaks in the spectrum 
Uevol=L.U(Dt=Dt)  #Propagator for the evolution time
# We could also use L.Sequence(Dt=Dt).U() above, for an empty sequence
Udelay=L.U(Dt=5) #Get the propagator for the delay (Here set to 5 seconds, i.e. full exchange)


# ## Execute the sequence
# We need to capture both an indirect and direct dimension, with both real and imaginary components. We'll loop over the indirect dimension, and capture the direct dimension with Rho.DetProp

# In[6]:


RE=list()
IM=list()
n=32
for k in range(n):
    rho.clear()  #Clear all data in rho
    #First capture the real part
    Uevol**k*rho  #This applies the evolution operator k times
    Upi2my*rho     #This flips the x magnetization up to z
    Udelay*rho     #This applies the delay for exchange to occur
    Upi2y*rho    #The flips the z magnetization back to x
    #You can write these all in one line instead: Upi2y*Udelay*Upi2my*Uevol**k*rho
    rho.DetProp(Uevol,n=n) #Detect the transverse (S0p) magnetization
    RE.append(rho.I[0])
    #Now capture the imaginary part
    rho.clear()
    Uevol**k*rho
    Upi2x*rho #Flip the y magnetization up to z
    Udelay*rho
    Upi2y*rho #Flip z magnetization back to x
    rho.DetProp(Uevol,n=n)
    IM.append(rho.I[0])


# ## Fourier transform and plot the results
# We just extract the real part from the spectrum.
# 
# SLEEPY has built-in functions for processing 1D spectra, but 2D processing must still be performed by the user.

# In[7]:


RE,IM=np.array(RE,dtype=complex),np.array(IM,dtype=complex) #Turn lists into arrays
# Divide first time points by zero
RE[:,0]/=2
RE[0,:]/=2
IM[:,0]/=2
IM[0,:]/=2
# QSINE apodization function
apod=np.cos(np.linspace(0,1,RE.shape[0])*np.pi/2)**2
RE=RE*apod
RE=(RE.T*apod).T
IM=IM*apod
IM=(IM.T*apod).T


# Hypercomplex processing

# In[8]:


nft=n*32
FT_RE=np.fft.fft(RE,n=nft,axis=1).real.astype(complex)
FT_IM=np.fft.fft(IM,n=nft,axis=1).real.astype(complex)
spec=np.fft.fftshift(np.fft.fft(FT_RE+1j*FT_IM,n=nft,axis=0),axes=[0,1])
v=1/(2*Dt)*np.linspace(-1,1,spec.shape[0])  #Frequency axis
v-=(v[1]-v[0])/2 #Shift to have zero at correct position
v*=1e6/ex0.v0[0]   #convert to ppm
vx,vy=np.meshgrid(v,v)  #meshgrid for plotting


# Plotting

# In[9]:


from matplotlib import cm
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(vx,vy,spec.real,cmap=cm.coolwarm,linewidth=0,color='None')
ax.set_xlabel(r'$\delta_1 (^{13}$C) / ppm')
ax.set_ylabel(r'$\delta_2 (^{13}$C) / ppm')
ax.invert_xaxis()
ax.invert_yaxis()

fig.set_size_inches([8,8])


# ## Sweep the delay time to observe buildup
# We just repeat the above code except with different lengths for Udelay. We slice through the larger peak in order to see the growth of the second peak

# In[10]:


delays=np.linspace(0,3,13)[:-1]
fig,ax=plt.subplots(3,4)
ax=ax.flatten()
sm=None
I=list()
for a,delay in zip(ax,delays):
    Udelay=L.U(Dt=delay)
    #Propagation
    RE=list()
    IM=list()
    for k in range(n):
        rho.clear()
        #First capture the real part
        Uevol**k*rho  #This applies the evolution operator k times
        Upi2my*rho     #This flips the x magnetization up to z
        Udelay*rho     #This applies the delay for exchange to occur
        Upi2y*rho    #The flips the z magnetization back to x
        rho.DetProp(Uevol,n=n) #Detect the transverse (S0p) magnetization
        RE.append(rho.I[0])
        #Now capture the imaginary part
        rho.clear()
        Uevol**k*rho
        Upi2x*rho #Flip the y magnetization up to z
        Udelay*rho
        Upi2y*rho #Flip z magnetization back to x
        rho.DetProp(Uevol,n=n)
        IM.append(rho.I[0])
    
    #Calculate spectrum
    RE,IM=np.array(RE,dtype=complex),np.array(IM,dtype=complex)
    RE[:,0]/=2
    RE[0,:]/=2
    IM[:,0]/=2
    IM[0,:]/=2
    RE=RE*apod
    RE=(RE.T*apod).T
    IM=IM*apod
    IM=(IM.T*apod).T
    spec=np.fft.fftshift(np.fft.fft(np.fft.fft(RE,n=nft,axis=1).real.astype(complex)+                                    1j*np.fft.fft(IM,n=nft,axis=1),n=nft,axis=0),axes=[0,1])
    v=1/(2*Dt)*np.linspace(-1,1,spec.shape[0])  #Frequency axis
    v-=(v[1]-v[0])/2  
    v*=1e6/ex0.v0[0]   #convert to ppm
    
    #Plot the result
    integral=spec[:,100:400].sum(1).real
    if sm is None:
        sm=integral.max()
    
    a.plot(v,integral)
    if a.is_last_row() if hasattr(a,'is_last_row') else a.get_subplotspec().is_last_row():
        a.set_xlabel(r'$\delta (^{13}$C) / ppm')
    if a.is_first_col() if hasattr(a,'is_first_col') else a.get_subplotspec().is_first_col():
        a.set_ylabel('I / a.u.')
    a.set_ylim([-sm*.3,sm*1.1])
    a.text(-10,-sm*.25,r'$\tau$'+f' = {delay:.1f} s')
    a.set_yticklabels('')
    
    I.append([spec[100:400][:,100:400].real.sum(),spec[600:900][:,600:900].real.sum(),
              spec[100:400][:,600:900].real.sum(),spec[600:900][:,100:400].real.sum()])  #Collect individual peak intensities
I=np.array(I).T   #Collection of individual peak intensities
fig.set_size_inches([9,7])
fig.tight_layout()


# ## Plot trajectory of the individual peaks
# Each peak represents the probability of starting in some state and ending in another state after the delay time, $\tau$

# In[11]:


ax=plt.subplots()[1]
ax.plot(delays,I.T)
ax.set_xlabel(r'$\tau$ / s')
ax.set_ylabel('I / a.u.')
ax.set_yticklabels('')
ax.legend((r'$p_1\rightarrow p_1$',r'$p_2\rightarrow p_2$',r'$p_1\rightarrow p_2$',r'$p_2\rightarrow p_1$'))


# ## Spectra as a function of exchange rate
# We can't use EXSY for faster motions, because the peaks don't stay separated. We can observe this behavior here. We just copy the setup from above for the 3D spectra and run it with varying correlation times

# In[12]:


tc0=np.logspace(-1,-6,6)
p1=0.5  #Population of state 1
p2=1-p1  #Population of state 2

fig=plt.figure()
ax=[fig.add_subplot(2,3,k+1,projection='3d') for k in range(6)]
I=list()
for a,tc in zip(ax,tc0):
    L.kex=1/(2*tc)*(np.array([[-1,1],[1,-1]])+(p1-p2)*np.array([[1,1],[-1,-1]]))
    
    Upi2x=pi2x.U()  #Get the propagators for the pulse sequence
    Upi2y=pi2y.U()
    Upi2my=pi2my.U()
    
    Udelay=L.U(Dt=tc0[0]*5)
    Uevol=L.U(Dt=Dt)  #Propagator for the evolution time
    
    #Propagation
    RE=list()
    IM=list()
    for k in range(n):
        rho.clear()
        #First capture the real part
        Uevol**k*rho  #This applies the evolution operator k times
        Upi2my*rho     #This flips the x magnetization up to z
        Udelay*rho     #This applies the delay for exchange to occur
        Upi2y*rho    #The flips the z magnetization back to x
        rho.DetProp(Uevol,n=n) #Detect the transverse (S0p) magnetization
        RE.append(rho.I[0])
        #Now capture the imaginary part
        rho.clear()
        Uevol**k*rho
        Upi2x*rho #Flip the y magnetization up to z
        Udelay*rho
        Upi2y*rho #Flip z magnetization back to x
        rho.DetProp(Uevol,n=n)
        IM.append(rho.I[0])
    
    #Calculate spectrum
    RE,IM=np.array(RE,dtype=complex),np.array(IM,dtype=complex)
    RE[:,0]/=2
    RE[0,:]/=2
    IM[:,0]/=2
    IM[0,:]/=2
    RE=RE*apod
    RE=(RE.T*apod).T
    IM=IM*apod
    IM=(IM.T*apod).T
    spec=np.fft.fftshift(np.fft.fft(np.fft.fft(RE,n=nft,axis=1).real+                                    1j*np.fft.fft(IM,n=nft,axis=1),n=nft,axis=0).real,axes=[0,1])
    v=1/(2*Dt)*np.linspace(-1,1,spec.shape[0])  #Frequency axis
    v-=(v[1]-v[0])/2  
    v*=1e6/ex0.v0[0]   #convert to ppm
    
    #Plot the result
    a.plot_surface(vx,vy,spec.real,cmap=cm.coolwarm,linewidth=0,color='None')
    a.set_xlabel(r'$\delta_1 (^{13}$C) / ppm',fontsize=8)
    a.set_ylabel(r'$\delta_2 (^{13}$C) / ppm',fontsize=8)
    a.tick_params(axis='both', which='major', labelsize=8)
    a.invert_xaxis()
    a.invert_yaxis()
    a.text(10,-10,a.get_zlim()[1]*1.3,r'$\tau_c = $'+f'{tc:.1e} s')
    
fig.set_size_inches(8,6)
fig.tight_layout()
for a in ax:a.set_zticklabels('')


# Now we do the same as above, but without having symmetric exchange, i.e. $p_1\ne p_2$.

# In[13]:


tc0=np.logspace(-1,-6,6)
p1=0.75  #Population of state 1
p2=1-p1  #Population of state 2

fig=plt.figure()
ax=[fig.add_subplot(2,3,k+1,projection='3d') for k in range(6)]
I=list()
for a,tc in zip(ax,tc0):
    L.kex=1/(2*tc)*(np.array([[-1,1],[1,-1]])+(p1-p2)*np.array([[1,1],[-1,-1]]))
    
    Upi2x=pi2x.U()  #Get the propagators for the pulse sequence
    Upi2y=pi2y.U()
    Upi2my=pi2my.U()
    
    Udelay=L.U(Dt=tc0[0]*5)
    Uevol=L.U(Dt=Dt)  #Propagator for the evolution time
    
    #Propagation
    RE=list()
    IM=list()
    for k in range(n):
        rho.clear()
        #First capture the real part
        Uevol**k*rho  #This applies the evolution operator k times
        Upi2my*rho     #This flips the x magnetization up to z
        Udelay*rho     #This applies the delay for exchange to occur
        Upi2y*rho    #The flips the z magnetization back to x
        rho.DetProp(Uevol,n=n) #Detect the transverse (S0p) magnetization
        RE.append(rho.I[0])
        #Now capture the imaginary part
        rho.clear()
        Uevol**k*rho
        Upi2x*rho #Flip the y magnetization up to z
        Udelay*rho
        Upi2y*rho #Flip z magnetization back to x
        rho.DetProp(Uevol,n=n)
        IM.append(rho.I[0])
    
    #Calculate spectrum
    RE,IM=np.array(RE,dtype=complex),np.array(IM,dtype=complex)
    RE[:,0]/=2
    RE[0,:]/=2
    IM[:,0]/=2
    IM[0,:]/=2
    RE=RE*apod
    RE=(RE.T*apod).T
    IM=IM*apod
    IM=(IM.T*apod).T
    spec=np.fft.fftshift(np.fft.fft(np.fft.fft(RE,n=nft,axis=1).real+                                    1j*(np.fft.fft(IM,n=nft,axis=1).real),n=nft,axis=0),axes=[0,1])
    v=1/(2*Dt)*np.linspace(-1,1,spec.shape[0])  #Frequency axis
    v-=(v[1]-v[0])/2  
    v*=1e6/ex0.v0[0]   #convert to ppm
    
    #Plot the result
    a.plot_surface(vx,vy,spec.real,cmap=cm.coolwarm,linewidth=0,color='None')
    a.set_xlabel(r'$\delta_1 (^{13}$C) / ppm',fontsize=8)
    a.set_ylabel(r'$\delta_2 (^{13}$C) / ppm',fontsize=8)
    a.tick_params(axis='both', which='major', labelsize=8)
    a.invert_xaxis()
    a.invert_yaxis()
    a.text(10,-10,a.get_zlim()[1]*1.3,r'$\tau_c = $'+f'{tc:.1e} s')
    

fig.set_size_inches(8,6)
fig.tight_layout()
for a in ax:a.set_zticklabels('')


# In[ ]:




