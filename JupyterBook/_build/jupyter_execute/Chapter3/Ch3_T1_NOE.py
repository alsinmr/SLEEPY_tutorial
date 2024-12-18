#!/usr/bin/env python
# coding: utf-8

# # <font  color = "#0093AF"> $T_1$ and NOE

# <a href="https://githubtocolab.com/alsinmr/SLEEPY_tutorial/blob/main/ColabNotebooks/Chapter3/Ch3_T1_NOE.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

# SLEEPY can simulate $T_1$ relaxation due to exchange dynamics, but requires being run in the lab frame. Modulation of a CSA will yield relaxation of one spin, whereas modulation of a dipole coupling will also bring about a polarization transfer between the spins (Nuclear Overhauser Effect).

# In[ ]:


# SETUP SLEEPY


# In[2]:


import SLEEPY as sl
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


ex0=sl.ExpSys(v0H=600,Nucs='13C',vr=0,LF=True,pwdavg=sl.PowderAvg())
ex0.set_inter('CSA',i=0,delta=200)
ex1=ex0.copy()
ex1.set_inter('CSA',i=0,delta=200,euler=[0,45*np.pi/180,0])

L=sl.Liouvillian(ex0,ex1)
L.kex=sl.Tools.twoSite_kex(1e-10)

seq=L.Sequence(Dt=1e-2)

rho=sl.Rho('Thermal','13Cz')

rho.DetProp(seq,n=1000)


# In[4]:


rho.plot()


# In[114]:


ex0=sl.ExpSys(v0H=600,Nucs=['13C','1H'],vr=0,LF=True,pwdavg=sl.PowderAvg()[10])
ex0.set_inter('dipole',i0=0,i1=1,delta=sl.Tools.dipole_coupling(.109,'1H','13C'))
ex1=ex0.copy()
ex1.set_inter('dipole',i0=0,i1=1,delta=sl.Tools.dipole_coupling(.109,'1H','13C'),euler=[0,45*np.pi/180,0])

L=sl.Liouvillian(ex0,ex1)
L.kex=sl.Tools.twoSite_kex(1e-12)

L.add_relax('DynamicThermal')

seq=L.Sequence(Dt=1e-4)
U=seq.U()**100

rho=sl.Rho('Thermal',['1Hz','13Cz'])
# L.Udelta('1H',np.pi)*rho

rho.DetProp(U,n=50000)


# In[115]:


rho.plot(axis='s')


# In[99]:


rho._detect[0]@((np.linalg.pinv(L[0].L(0))@L[0].L(0))@L.rho_eq(sub1=True))


# In[41]:


n=np.argmax(U._eig[0][0].real)
v=U._eig[0][1]
vi=np.linalg.pinv(v)
mat=np.atleast_2d(v[:,n]).T@np.atleast_2d(vi[n])


# In[42]:


(rho._detect[0]@(mat@rho._rho0[0]))


# In[43]:


ex0.Peq


# In[44]:


(np.abs(U._eig[0][0])>1e-10).sum()


# In[45]:


U._eig[0][0]


# In[48]:


U._eig[0][0][12]


# In[ ]:




