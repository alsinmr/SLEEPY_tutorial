#!/usr/bin/env python
# coding: utf-8

# # <font  color = "#0093AF"> T$_1$ and NOE

# In[1]:


import os
os.chdir('../../../')
import SLEEPY as sl
import numpy as np
import matplotlib.pyplot as plt
from time import time


# In[2]:


ex0=sl.ExpSys(v0H=600,Nucs='13C',vr=0,LF=True,pwdavg=sl.PowderAvg())
ex0.set_inter('CSA',i=0,delta=200)
ex1=ex0.copy()
ex1.set_inter('CSA',i=0,delta=200,euler=[0,45*np.pi/180,0])

L=sl.Liouvillian(ex0,ex1)
L.kex=sl.Tools.twoSite_kex(1e-10)

seq=L.Sequence(Dt=1e-2)

rho=sl.Rho('Thermal','13Cz')

rho.DetProp(seq,n=1000)


# In[3]:


rho.plot()


# In[4]:


ex0=sl.ExpSys(v0H=600,Nucs=['13C','1H'],vr=0,LF=True,pwdavg=sl.PowderAvg()[0])
ex0.set_inter('dipole',i0=0,i1=1,delta=sl.Tools.dipole_coupling(.209,'1H','13C'))
ex1=ex0.copy()
ex1.set_inter('dipole',i0=0,i1=1,delta=sl.Tools.dipole_coupling(.209,'1H','13C'),euler=[0,45*np.pi/180,0])

L=sl.Liouvillian(ex0,ex1)
L.kex=sl.Tools.twoSite_kex(1e-10)

L.add_relax('DynamicThermal')

seq=L.Sequence(Dt=1e-2)

rho=sl.Rho('Thermal',['1Hz','13Cz'])

rho.DetProp(seq,n=50000)


# In[5]:


rho.plot(axis='s')


# In[6]:


L[0].L(step=0)@L[0].rho_eq(step=0)


# In[7]:


U=L.Sequence(Dt=1e-3).U()


# In[8]:


a,b=np.linalg.eig(U[0])


# In[9]:


b[:,14]/L.rho_eq(step=0)


# In[ ]:




