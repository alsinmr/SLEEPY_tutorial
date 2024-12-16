#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('../../../')
import SLEEPY as sl
import numpy as np
import matplotlib.pyplot as plt
from time import time


# <a href="https://githubtocolab.com/alsinmr/SLEEPY_tutorial/blob/main/ColabNotebooks/Chapter4/Ch4_CrossEffect.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

# In[2]:


ex=sl.ExpSys(v0H=211,Nucs=['e','14N'],vr=0,T_K=80,pwdavg=sl.PowderAvg('zcw986'))
print(ex.pwdavg.n_gamma)
ex.set_inter('g',i=0,gxx=2.0022,gyy=2.0059,gzz=2.0098,euler=[0,0,0])
ex.set_inter('hyperfine',i0=0,i1=1,Axx=20e6,Ayy=20e6,Azz=100e6)

L=ex.Liouvillian()
L.add_relax('T2',i=0,T2=400e-9)
L.add_relax('T1',i=0,T1=500e-6)

seq=L.Sequence(Dt=1e-9)
rho=sl.Rho('ex','ep')
rho.DetProp(seq,n=256)
rho.apod_pars['LB']=40e6
rho.plot(FT=True,apodize=True,axis='MHz')


# In[3]:


ex=sl.ExpSys(v0H=250,Nucs=['e-','e-','1H'],LF=[False,False,True],vr=5000,T_K=80,pwdavg=sl.PowderAvg(q=2))
ex.set_inter('hyperfine',i0=1,i1=2,Axx=-1000000,Ayy=-1000000,Azz=2000000,euler=[0,0,0])
ex.set_inter('hyperfine',i0=0,i1=2,Axx=-1000000,Ayy=-1000000,Azz=2000000,euler=[0,np.pi/2,0])
ex.set_inter('g',i=0,gxx=2.0022,gyy=2.0059,gzz=2.0098,euler=[0,0,0])
ex.set_inter('g',i=1,gxx=2.0022,gyy=2.0059,gzz=2.0098,euler=[0,np.pi/2,0])
ex.set_inter('dipole',i0=0,i1=1,delta=sl.Tools.dipole_coupling(1.33,'e-','e-'),euler=[0,77*np.pi/180,0])

L=ex.Liouvillian()
L.add_relax('T2',i=0,T2=5e-6)
L.add_relax('T2',i=1,T2=5e-6)
L.add_relax('T1',i=0,T1=250e-6)
L.add_relax('T1',i=1,T1=250e-6)
L.add_relax('T2',i=2,T2=5e-3)
L.add_relax('T1',i=2,T1=3)
L.add_relax('recovery')

seq=L.Sequence().add_channel('e-',v1=1e6,voff=-250e6)

ON=sl.Rho('Thermal',['S0z','S1z','1Hz'])
ON.DetProp(seq,n=20000)
ON.plot(axis='s',det_num=2)


# In[17]:


seq=L.Sequence()

OFF=sl.Rho('Thermal',['S0z','S1z','1Hz'])
OFF.DetProp(seq,n=20000)
OFF.plot(axis='s',det_num=2)


# In[18]:


print(f'Apparent enhancement: {(ON.I[-1][-1]/OFF.I[-1][-1]).real:.0f}')
print(f'Real enhancement: {(ON.I[-1][-1]/ex.Peq[-1]).real:.0f}')


# In[14]:


fig=plt.figure()
ax=[fig.add_subplot(10,5,k+1) for k in range(len(ON.Ipwd))]
for k,a in enumerate(ax):
    a.plot(ON.t_axis[::100],ON.Ipwd[k][-1][::100].real)
    a.set_ylim([ON.Ipwd[:,-1].real.min(),ON.Ipwd[:,-1].real.max()])
    if a.is_first_col():
        a.set_ylabel(r'P($^1$H)')
    else:
        a.set_yticklabels([])
    if a.is_last_row():
        a.set_xlabel('t / s')
    else:
        a.set_xticklabels([])
fig.set_size_inches([8,12])
fig.tight_layout()


# In[22]:


ex=sl.ExpSys(v0H=250,Nucs=['e-','e-','1H'],LF=[False,False,True],vr=5000,T_K=80,pwdavg=sl.PowderAvg(q=2)[45])
ex.set_inter('hyperfine',i0=1,i1=2,Axx=-1000000,Ayy=-1000000,Azz=2000000,euler=[0,0,0])
ex.set_inter('hyperfine',i0=0,i1=2,Axx=-1000000,Ayy=-1000000,Azz=2000000,euler=[0,np.pi/2,0])
ex.set_inter('g',i=0,gxx=2.0022,gyy=2.0059,gzz=2.0098,euler=[0,0,0])
ex.set_inter('g',i=1,gxx=2.0022,gyy=2.0059,gzz=2.0098,euler=[0,np.pi/2,0])
ex.set_inter('dipole',i0=0,i1=1,delta=sl.Tools.dipole_coupling(1.33,'e-','e-'),euler=[0,77*np.pi/180,0])

L=ex.Liouvillian()
L.add_relax('T2',i=0,T2=5e-6)
L.add_relax('T2',i=1,T2=5e-6)
L.add_relax('T1',i=0,T1=250e-6)
L.add_relax('T1',i=1,T1=250e-6)
L.add_relax('T2',i=2,T2=5e-3)
L.add_relax('T1',i=2,T1=3)
L.add_relax('recovery',OS=True)

seq=L.Sequence().add_channel('e-',v1=1e6,voff=-250e6)

fig,ax=plt.subplots(1,3,sharex=True)

Op=ex.Op

rho=sl.Rho('Thermal',['S0z','S1z',Op[0].p@Op[2].m+Op[0].m@Op[2].p,
                      Op[1].p@Op[2].m+Op[1].m@Op[2].p,'1Hz'])
rho.clear()
rho.DetProp(seq,n=300,n_per_seq=100)
rho.plot(axis='us',ax=ax[0],det_num=[0,1])
rho.plot(axis='us',ax=ax[1],det_num=[2,3])
rho.plot(axis='us',ax=ax[2],det_num=4)
fig.set_size_inches([9,3.5])
fig.tight_layout()


# In[23]:


ex=sl.ExpSys(v0H=250,Nucs=['e-','e-','1H'],LF=[False,False,True],vr=5000,T_K=80,pwdavg=sl.PowderAvg(q=2)[45])
ex.set_inter('hyperfine',i0=1,i1=2,Axx=-1000000,Ayy=-1000000,Azz=2000000,euler=[0,0,0])
ex.set_inter('hyperfine',i0=0,i1=2,Axx=-1000000,Ayy=-1000000,Azz=2000000,euler=[0,np.pi/2,0])
ex.set_inter('g',i=0,gxx=2.0022,gyy=2.0059,gzz=2.0098,euler=[0,0,0])
ex.set_inter('g',i=1,gxx=2.0022,gyy=2.0059,gzz=2.0098,euler=[0,np.pi/2,0])
ex.set_inter('dipole',i0=0,i1=1,delta=sl.Tools.dipole_coupling(1.33,'e-','e-'),euler=[0,77*np.pi/180,0])

L=ex.Liouvillian()
L.add_relax('T2',i=0,T2=5e-6,OS=True)
L.add_relax('T2',i=1,T2=5e-6,OS=True)
L.add_relax('T1',i=0,T1=250e-6,OS=True,Thermal=True)
L.add_relax('T1',i=1,T1=250e-6,OS=True,Thermal=True)
L.add_relax('T2',i=2,T2=5e-3,OS=True)
L.add_relax('T1',i=2,T1=3,OS=True,Thermal=True)

seq=L.Sequence().add_channel('e-',v1=1e6,voff=-250e6)

fig,ax=plt.subplots(1,3,sharex=True)

Op=ex.Op

rho=sl.Rho('Thermal',['S0z','S1z',Op[0].p@Op[2].m+Op[0].m@Op[2].p,
                      Op[1].p@Op[2].m+Op[1].m@Op[2].p,'1Hz'])
rho.clear()
rho.DetProp(seq,n=300,n_per_seq=100)
rho.plot(axis='us',ax=ax[0],det_num=[0,1])
rho.plot(axis='us',ax=ax[1],det_num=[2,3])
rho.plot(axis='us',ax=ax[2],det_num=4)
fig.set_size_inches([9,3.5])
fig.tight_layout()


# In[5]:


ex=sl.ExpSys(v0H=250,Nucs=['e-','e-','1H'],LF=[False,False,True],vr=5000,T_K=80,
             pwdavg=sl.PowderAvg(q=2)[10],rotor_angle=np.arccos(np.sqrt(1/3)))
ex.set_inter('hyperfine',i0=1,i1=2,Axx=-1000000,Ayy=-1000000,Azz=2000000,euler=[0,0,0])
ex.set_inter('hyperfine',i0=0,i1=2,Axx=-1000000,Ayy=-1000000,Azz=2000000,euler=[0,np.pi/2,0])
ex.set_inter('g',i=0,gxx=2.0022,gyy=2.0059,gzz=2.0098,euler=[0,0,0])
ex.set_inter('g',i=1,gxx=2.0022,gyy=2.0059,gzz=2.0098,euler=[0,np.pi/2,0])
ex.set_inter('dipole',i0=0,i1=1,delta=sl.Tools.dipole_coupling(1.33,'e-','e-'),euler=[0,77*np.pi/180,0])

Dt=1e-3 if ex.vr==0 else None

L=ex.Liouvillian()
L.add_relax('T2',i=0,T2=5e-6)
L.add_relax('T2',i=1,T2=5e-6)
L.add_relax('T1',i=0,T1=250e-6)
L.add_relax('T1',i=1,T1=250e-6)
L.add_relax('T2',i=2,T2=5e-3)
L.add_relax('T1',i=2,T1=3)
L.add_relax('Thermal')

Op=ex.Op
rho=sl.Rho('Thermal',['S0z','S1z',Op[0].p@Op[1].m@Op[2].p+Op[0].m@Op[1].p@Op[2].p,
                      1000*Op[0].z@Op[1].z,Op[0].z+Op[1].z+Op[2].z,'1Hz'])

voff=-100e6
seq=L.Sequence(Dt).add_channel('e-',v1=1e6,voff=voff)

fig,ax=plt.subplots(2,3,sharex=True)
print(np.abs(np.linalg.eig(seq.U(Dt)[0])[0]).max())



rho.clear()
rho.DetProp(seq,n=300,n_per_seq=100)
rho.downmix()
rho.plot(axis='us',ax=ax[0,0],det_num=[0,1,4])
rho.plot(axis='us',ax=ax[0,1],det_num=[2,3])
rho.plot(axis='us',ax=ax[0,2],det_num=-1)
fig.set_size_inches([9,3.5])
fig.tight_layout()

seq=L.Sequence(Dt).add_channel('e-',v1=0*1e6,voff=voff)
print(np.abs(np.linalg.eig(seq.U(Dt)[0])[0]).max())

rho.clear()
rho.DetProp(seq,n=300,n_per_seq=100)
rho.downmix()
rho.plot(axis='us',ax=ax[1,0],det_num=[0,1,4])
rho.plot(axis='us',ax=ax[1,1],det_num=[2,3])
rho.plot(axis='us',ax=ax[1,2],det_num=-1)
fig.set_size_inches([9,5])
for a in ax[0]:a.set_xlabel('')
fig.tight_layout()


# In[ ]:




