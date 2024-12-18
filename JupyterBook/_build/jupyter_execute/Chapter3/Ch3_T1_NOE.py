#!/usr/bin/env python
# coding: utf-8

# # <font  color = "#0093AF"> $T_1$ and NOE

# <a href="https://githubtocolab.com/alsinmr/SLEEPY_tutorial/blob/main/ColabNotebooks/Chapter3/Ch3_T1_NOE.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

# SLEEPY can simulate $T_1$ relaxation due to exchange dynamics, but requires being run in the lab frame. Modulation of a CSA will relaxation one spin, whereas modulation of a dipole coupling will also bring about a polarization transfer (Nuclear overhauser effect).

# In[ ]:


# SETUP SLEEPY


# In[2]:


import SLEEPY as sl
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


ex0=sl.ExpSys(v0H=600,Nucs='13C',vr=0,LF=True,pwdavg=sl.PowderAvg())
ex0.set_inter('CSA',i=0,delta=200)
ex1=ex0.copy()
ex1.set_inter('CSA',i=0,delta=200,euler=[0,45*np.pi/180,0])

L=sl.Liouvillian(ex0,ex1)
L.kex=sl.Tools.twoSite_kex(1e-10)

seq=L.Sequence(Dt=1e-2)

rho=sl.Rho('Thermal','13Cz')

rho.DetProp(seq,n=1000)


# In[6]:


rho.plot()


# In[5]:


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


# In[6]:


rho.plot(axis='s')


# In[8]:


L[0].L(step=0)@L[0].rho_eq(step=0)


# In[10]:


U=L.Sequence(Dt=1e-3).U()


# In[11]:


a,b=np.linalg.eig(U[0])


# In[15]:


b[:,14]/L.rho_eq(step=0)


# In[ ]:




