#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 21:52:56 2025

@author: felipe
"""

import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import h5py
from calc_eigendecomposition import calc_eigendecomposition, calc_parcellate, calc_normalize_timeseries
from nilearn import plotting
from nilearn import surface
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import laplacian
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import eigsh, inv,lsqr
from nilearn.surface import SurfaceImage
import matplotlib.pyplot as plt
import gc



# plt.plot(f,avg_Pxx)
LeftHemisphere=surface.load_surf_mesh('/home/felipe/Dropbox/VIBEBRAIN/NetworkSize/L_hemisphere_template.gii')
coordinates_L=LeftHemisphere.coordinates

N_total=np.shape(coordinates_L)[0]

#Remove the interhemisphere wall
data_cortex = np.genfromtxt('/home/felipe/Dropbox/VIBEBRAIN/NetworkSize/fsLR_32k_cortex-lh_mask.txt', delimiter='\n')
ind_cortex=np.argwhere(data_cortex==1)
N_cortex=np.shape(ind_cortex)[0]
coordinates_cortex=coordinates_L[ind_cortex[:,0]]
distance_matrix=cdist(coordinates_cortex,coordinates_cortex,metric='euclidean')
distance_matrix*=0.001 #just translate to meters


del LeftHemisphere
del coordinates_L, coordinates_cortex
del data_cortex

gc.collect()
#%%
print('Calculating C')
C=np.exp(-60*distance_matrix) #EDR 120 by Pang 2023
C[np.eye(np.shape(C)[0])==1]=0
C=np.where(C<0.1,0,C)
C=coo_matrix(C)
print(100*len(C.nonzero()[0])/N_cortex**2)

#%%
print('Loading timeseries')
#Load fMRI data
file_series=h5py.File('/home/felipe/Descargas/subject_rfMRI_timeseries-lh.mat','r')
timeseries=file_series['timeseries']
timeseries=np.array(timeseries[:])

fs=1/0.72
f,Pxx=signal.welch(timeseries.T,fs=fs,nperseg=200,noverlap=100)

avg_Pxx=np.mean(Pxx[ind_cortex,:],axis=0)
f_peak=f[np.argmax(avg_Pxx)]
f_peaks=f[np.argmax(Pxx[:,:],axis=1)[ind_cortex]]
#%%
MDs=np.arange(0.08,0.091,0.01)
mean_Omega=np.zeros((len(MDs),))
error=np.zeros((len(MDs),))
print('Calculating Omega')

for m,MD in enumerate(MDs):#[200,500,1000,2000,5000,10000]
    print(m)    
    tau=(C!=0).multiply(distance_matrix)*MD
    K=N_cortex*5
    A=-(C.multiply(tau)+laplacian(C))*K/N_cortex
    omega=np.ones((N_cortex))*0.1
    Omega_tuple=lsqr(A=identity(N_cortex)-A,b=omega,iter_lim=500,show=True,x0=(omega/(1+MD)))
    mean_Omega[m]=np.nanmean(Omega_tuple[0])
    error[m]=np.sum(np.abs(Omega_tuple[0]-f_peaks))

#%%
plt.subplot(1,2,1)
plt.plot(MDs,mean_Omega[0:3],':o')
plt.hlines(np.mean(f_peaks),xmin=0,xmax=0.07)

plt.subplot(1,2,2)
plt.plot(MDs,error[0:3],':o')
#%%
MD=0.1
tau=(C!=0).multiply(distance_matrix)*MD
K=N_cortex*5
A=-(C.multiply(tau)+laplacian(C))*K/N_cortex
omega=np.ones((N_cortex))
Omega_tuple=lsqr(A=identity(N_cortex)-A,b=omega,iter_lim=500,show=True,x0=(omega/(1+MD)))
Omega=Omega_tuple[0]
#%%
plt.plot(f_peaks[0:1000])
plt.plot(Omega[0:1000])

cc=np.corrcoef(np.reshape(f_peaks,[1,-1]),np.reshape(Omega,[1,-1]))
