#!/usr/bin/env python
# coding: utf-8

# # <font  color = "#0093AF"> Hamiltonians and Liouvillians

# <a href="https://githubtocolab.com/alsinmr/SLEEPY_tutorial/blob/main/ColabNotebooks/Chapter1/Ch1_Liouvillian.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

# ## Setup

# In[1]:


import os
os.chdir('../../../')
import SLEEPY as sl
import numpy as np


# ## The Hamiltonian

# In the previous section, we defined the experimental conditions and the spin-system. The 'ex' object, created with sl.ExpSys, contains all the information required to build the Hamiltonian and Liouvillian (for exchange, we need multiple 'ex' objects).
# 
# SLEEPY operates in Liouville space, so while the Hamiltonian is in the background, the user will normally create the Liouvillian directly. Still, here we will start with the Hamiltonian and its contents.
# 
# We start with building 'ex'.

# In[2]:


ex=sl.ExpSys(v0H=600,Nucs=['1H','13C'],vr=60000,LF=True)  #For this example, we'll work in the lab frame
ex.set_inter('dipole',i0=0,i1=1,delta=44000).set_inter('CSA',i=1,delta=100,eta=1).    set_inter('CS',i=0,ppm=10) #Add a dipole, CSA to 13C, and CS to 1H

H=ex.Hamiltonian() #Create the Hamiltonian


# To access Hamiltonian matrices, we have to specify which element of the powder average we are interested in, and also what step (up to ex.n_gamma) in the rotor cycle we want. For solution-state simulations, these are still required, but can both be set to zero. For example, for the first element of the powder average, and the first step in the rotor cycle, we use

# ### Access and plot the Hamiltonian

# In[3]:


H[0].H(0)


# Note that we have set up in the lab frame, explaining why the Hamiltonian is so dense. In the rotating frame, this will not be the case.
# 
# We may also visualize the Hamiltonian, using H.plot. Plotting may be performed for a given element of the powder average, or if not specified (but required)

# In[4]:


H[5].plot() #Plot the Hamiltonian for the 5th element of the powder average


# The Hamiltonian is built from the 5 rotating components of the Hamiltonian. If we have $n_\gamma$ steps per rotor period, and we're at the kth step, this is done as follows:
# 
# $$
# \begin{eqnarray}
# \phi&=&\exp(2\pi i k/n_\gamma) \nonumber \\
# \hat{H}&=&\phi^{-n}*\hat{H}_n
# \end{eqnarray}
# $$
# 
# The rotating components are obtained via the Hn function, where n (-2,-1,0,1,2) must be provided. For example, for the fifth element of the powder average, if we want the $n=-1$ component, we would call:

# In[5]:


H[5].Hn(-1)


# This may also be plotted. Note there are different plotting modes: 're','im','abs', 'log', and 'spy', which are real, imaginary, absolute value, log of absolute value, and spy, which is binary, i.e. zero or not zero.

# In[6]:


H[5].plot('H-1',mode='re')


# ### Hamiltonians for individual interactions

# The rotating components are built up from the rotating components of the individual interactions, where individual interactions are found in a list in H.Hinter. Indexing this list will return a description of the interaction, for example:

# In[7]:


H.Hinter[0]


# In[8]:


H.Hinter[-1]


# The Hamiltonians are those specified when using ex.set_inter. However, if a spin is specified in the lab frame, then we will also find Hamiltonians that add the Larmor frequency for those spins. As with the full Hamiltonian, we may plot Hamiltonians for the individual interactions.

# In[9]:


H.Hinter[0].plot()


# ## The Liouvillian

# Coherent components of the Liouvillian are calculated from the Hamiltonian, although we may also add relaxation and exchange processes to the Liouvillian. We may create the Liouvillian without exchange either from the Hamiltonian or the experimental system (ex). 

# In[10]:


L=ex.Liouvillian()  
L=sl.Liouvillian(H)  #alternatively


# The coherent Liouvillian is created by calculating
# 
# $$
# \begin{equation}
# \hat{\hat{L}}=i(\hat{H}\otimes\hat{\mathbf{1}}-\hat{\mathbf{1}}\otimes\hat{H})
# \end{equation}
# $$
# 
# $\hat{\mathbf{1}}$ is an identity matrix with the same dimensions as $\hat{H}$. In SLEEPY, we separate the individual components, such that we acquired
# 
# $$
# \begin{equation}
# \hat{\hat{L}}_n=i(\hat{H}_n\otimes\hat{\mathbf{1}}-\hat{\mathbf{1}}\otimes\hat{H}_n)
# \end{equation}
# $$

# ### Access and plot the Liouvillian

# Access to specific components of the Liouvillian and plotting use the same rules as for the Hamiltonian. For example, to access the 5th element of the powder average, and the $n=1$ component of the coherent Liouvillian, we would use
# ```
# L[5].Ln(1)
# ```
# Note that the $n=0$ component contains contributions from relaxation (non-orientation specific) and exchange. To obtain the pure coherent contributions, we use
# ```
# L[5].Ln_H(1)
# ```
# The full Liouvillian for a given element of the powder average and step in the rotor period is obtained via:
# ```
# L[5].L(10)  #5th powder average element, 10th step
# ```
# We may also plot these matrices, which is particularly helpful since the Liouvillian gets relatively large and numerical display of the full matrix may not be very helpful.

# In[11]:


L[0].plot()  #Full Liouvillian


# In[12]:


L[0].plot('L-1') #n=-1 component of the coherent Liouvillian


# ### Adding relaxation

# $T_1$ and $T_2$ relaxation are available in SLEEPY, along with "Spin-Diffusion" which just introduces a uniform signal decay in x,y, and z directions. As implemented $T_1$ *only* decays along $z$, which is unphysical, so it is important to also add some $T_2$ relaxation when using $T_1$. By default, $T_1$ acts along $z$, and $T_2$ along x and y. However, it is possible to specify "orientation-specific" (OS) relaxation, which will adjust $T_1$ relaxation to occur on the eigenstates of a given spin, and $T_2$ acts on coherences between eigenstates of the spin. This option is more computationally expensive. However, for spins that are strongly tilted away from $z$, using the default relaxation will "mix" the $T_1$ and $T_2$ behavior, so that $T_1$ may appear much shorter than specified. 
# 
# We may also include recovery of magnetization to thermal equilibrium (noting that in this case, ex.T_K becomes relevant). If orientation-specific relaxation is included, this is specified when the relaxation is added. Otherwise, it is specified after all relaxation is included. Relaxation is removed by running L.clear_relax. If, for example, $T_1$ relaxation is added to a spin twice without clearing the first entry, then the relaxation rates will add together, so it is important not to forget to clear existing relaxation.

# In[13]:


L.clear_relax()
L.add_relax('T1',i=0,T1=1)  #Add T1 relaxation to spin 0
L.add_relax('T2',i=0,T2=.1) #Add T2 relaxation to spin 0
L.add_relax('T1',i=1,T1=5)  #Add T1 relaxation to spin 0
L.add_relax('T2',i=1,T2=.5) #Add T2 relaxation to spin 0
_=L.add_relax('recovery') #Add recovery to thermal equilibrium


# To add orientation-specific relaxation, specify `OS=True`. To include recovery to thermal equilibrium, the $T_1$ require specifying `Thermal=True`.

# In[14]:


L.clear_relax()
L.add_relax('T1',i=0,T1=1,OS=True,Thermal=True)  #Add T1 relaxation to spin 0
L.add_relax('T2',i=0,T2=.1,OS=True) #Add T2 relaxation to spin 0
L.add_relax('T1',i=1,T1=5,OS=True,Thermal=True)  #Add T1 relaxation to spin 0
_=L.add_relax('T2',i=1,T2=.5,OS=True) #Add T2 relaxation to spin 0


# We can visualize the result with the plot function.

# In[15]:


L.plot('Lrelax',mode='abs')


# ### Exchange

# Importantly, SLEEPY allows us to simulate magnetic resonance under exchange conditions. This is achieved by defining two or more experimental system objects, with different interaction conditions. In this example, we just change the orientation of the dipole coupling, representing some kind of hopping motion.
# 
# Since the resulting Liouvillian comes from multiple experimental systems, we must use sl.Liouvillian, rather than generating it directly from ex.

# In[16]:


ex1=ex.copy() #We can copy an existing ex, so that to start, all parameters match
ex1.set_inter('dipole',i0=0,i1=1,delta=44000,euler=[0,30*np.pi/180,0]) #30 degree hop

L=sl.Liouvillian(ex,ex1)


# We can plot the resulting Liouvillian, to see that it now has larger dimension, corresponding to the two sets of conditions.

# In[17]:


L.plot()


# However, on its own, this isn't very useful because the two dipole orientations are not exchanging. For this, we must also include the exchange matrix, of the form:
# 
# $$
# \begin{equation}
# \mathbf{K}_{ex}=
# \begin{pmatrix}
# -k_{01}&k_{10} \\
# k_{01}&-k_{10}
# \end{pmatrix}
# \end{equation}
# $$
# 
# Exchange matrices should always be mass conserving, i.e., the columns should sum to 0. Usually, they should also satisfy detailed balance, i.e., $k_{m,n}/k_{n,m}=p_n/p_m$ where $p_n$ are equilibrium populations.

# In[18]:


L.kex=[[-1e3,1e3],[1e3,-1e3]]  #Symmetric exchange between two sites


# A number of tools exist in sl.Tools for building various exchange matrices, and also setting up both the exchange matrix and experimental systems (twoSite_kex,nSite_sym,fourSite_sym,Setup3siteSym,SetupTetraHop).
# 
# Below, we show the exchange component of the Liouvillian

# In[19]:


L.plot('Lex',mode='re')


# Finally, we show the total Liouvillian, including exchange.

# In[20]:


L.plot()


# In the next section, we discuss the generation of propagators from the Liouvillian.

# In[ ]:




