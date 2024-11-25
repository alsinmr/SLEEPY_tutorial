#!/usr/bin/env python
# coding: utf-8

# # <font  color = "#0093AF"> Experimental Settings and Spin-System Definition

# ## Setup

# In[1]:


import os
os.chdir('../../../')
import SLEEPY as sl
import numpy as np


# ## Defining nuclei and experimental conditions

# The experimental system defines the magnetic field, the nuclei in the spin-system, the spinning rate, the temperature, the rotor angle, the powder average, and the number of gamma angles calculated during one rotor period. Except for the field and nuclei, these all have default values and only need to be provided to change the default values.
# 
# - v0H: The magnetic field strength, given as the $^1$H frequency in MHz (required, unless B0 provided)
# - B0: The magnetic field strength in Tesla (required, unless v0H provided)
# - Nucs: List of nuclei, with mass number followed by atomic symbol ('1H','13C','2H', etc.). Electrons may also be included via 'e-'. Specifying 'e1' would give an electron with spin 1, and 'e3/2' or 'e1.5' would produce an electron with spin-3/2.
# - T_K: Temperature in Kelvin. Only used if relaxation to thermal equilibrium is used (thermalization), or the density matrix (rho) is initialized with the "thermal" option.
# - vr: Spinning frequency in Hz (only used if anisotropic interactions provided). Default is 10000
# - rotor_angle: Rotor angle, in radians. Default is the magic angle
# - n_gamma: Number of gamma angles calculated per rotor period. For string-specified powder averages, this is also the number of gamma angles in the powder average. Default is 100
# - pwdavg: Type of powder average. Type sl.PowderAvg.list_powder_types to see options (Most powder averages from SIMPSON). If an integer is provided, then this yields the JCP59 powder average, with higher integers yielding more angles. Defaults is 3 (JCP59 with 99 angles)
# - LF: Specifiy whether each spin should be simulating in the lab frame. Can be provided as a single boolean, e.g. False sets all spins in the rotating frame, or as a list the same length as Nucs, which puts some spins in the lab frame and some in the rotating frame (useful, e.g. for DNP experiments such as solid-effect, where the electron should be in the rotating frame, but the nucleus in the lab frame).

# In[2]:


ex=sl.ExpSys(v0H=600,Nucs=['1H','13C'],vr=10000,T_K=298,
             rotor_angle=np.arccos(np.sqrt(1/3)),n_gamma=100,
             pwdavg=3,LF=[False,False])


# Typing `ex` at the command line will return a description of the spin-system.

# Note that we have used the default values, so the same system may be obtained while omitting all the defaults:

# In[3]:


ex=sl.ExpSys(v0H=600,Nucs=['1H','13C'])


# ## Defining Interactions
# Once the experimental settings and spin-system is set, we may add interactions. This is achieved by running
# ```
# ex.set_inter(...)
# ```
# For every interaction, we have to specify the spins involved. For an N-spin system, this is specified with an index (spin-field) or indices (spin-spin) referring to the spin at the corresponding position in Nucs. Note we use python convention of indexing from 0 to N-1. For spin-field interactions, we specify "i", and for spin-spin interactions, we specify "i0" and "i1". The available interactions are:
# 
# - dipole: Specify delta (the full anisotropy in Hz, which is 2x the definition used by SIMPSON). Optionally specify an asymmetry, eta (unitless) and the euler angles, euler as a 3-element (alpha,beta,gamma) list in radians.
# - J: Specify J in Hz.
# - CS: Isotropic chemical shift, specify in ppm.
# - CSA: Chemical shift anisotropy. Specify delta in ppm. eta and the euler angles are optional.
# - hyperfine: Specify Axx, Ayy, and Azz. If all entries are equal, will be treated as an isotropic interaction. "euler" may be optionally provided.
# - quadrupole: Specify delta in Hz (CHECK THIS INPUT). Optionally specify eta and euler
# - g: Electron g-tensor. Specify gxx, gyy, and gzz, and optionally euler.
# 

# In[4]:


delta=sl.Tools.dipole_coupling(.105,'1H','13C')  #Calculate H-C dipole for 1.05 Angstrom distance
ex.set_inter('dipole',i0=0,i1=1,delta=delta) #H-C dipole coupling
ex.set_inter('CSA',i=1,delta=100,eta=1) #13C CSA
_=ex.set_inter('CS',i=0,ppm=10) #1H isotropic chemical shift


# Note that when setting an interaction, ex returns itself. This lets us string together multiple commands, for example, the following line will achieve the same interactions as above.

# In[5]:


_=ex.set_inter('dipole',i0=0,i1=1,delta=delta).set_inter('CSA',i=1,delta=100,eta=1).    set_inter('CS',i=0,ppm=10)


# If we just type 'ex' at the command line, we will obtain a description of the experimental system

# In[6]:


ex


# In[ ]:




