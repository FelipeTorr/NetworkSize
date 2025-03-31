#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:12:41 2025

@author: felipe
"""

import os
import numpy as np
import scipy.io as sio
import h5py
from calc_eigendecomposition import calc_eigendecomposition, calc_parcellate, calc_normalize_timeseries

#Load data of geometrical cortical eigenmodes from surface
file=np.load('LaplaceBeltramiEig.npz')
evals=file['evals']
evec=file['evec']

#Load data of geometrical-topology cortical eigenmodes
# file1=np.load('tau10000_eigenmodes.npz')
file1=np.load('tau10000_EDR_eigenmodes.npz')
evals1=file1['evals']
evec1=file1['evec']
evec1=evec1/np.max(evec1)

#Load fMRI data
file_series=h5py.File('/home/felipe/Descargas/subject_rfMRI_timeseries-lh.mat','r')
timeseries=file_series['timeseries']
timeseries=np.array(timeseries[:])

T=np.shape(timeseries)[0]
num_vertex=np.shape(evec)[0]
num_modes=np.shape(evec)[1]

#Valid indexes of the cortex
data_cortex = np.genfromtxt('/home/felipe/Dropbox/VIBEBRAIN/NetworkSize/fsLR_32k_cortex-lh_mask.txt', delimiter='\n')
cortex_ind=np.argwhere(data_cortex==1)[:,0]

#Indexes from surface to parcellation
data_parcellation=np.genfromtxt('/home/felipe/Dropbox/VIBEBRAIN/NetworkSize/fsLR_32k_Schaefer200-lh.txt')
num_parcels = len(np.unique(data_parcellation[data_parcellation>0]))
#%%
#Reconstruction coeficients for cortical eigenmodes
recon_beta = np.zeros((num_modes, T, num_modes));
for mode in range(num_modes):
    basis = evec[cortex_ind, 0:mode+1]
    recon_beta[0:mode+1,:,mode] = (calc_eigendecomposition(timeseries[:,cortex_ind], basis)).T
#%%
#Reconstruction for cortical-topological eigenmodes
recon_beta1 = np.zeros((num_modes, T, num_modes));
for mode in range(num_modes):
    basis = evec1[cortex_ind, 0:mode+1]
    recon_beta1[0:mode+1,:,mode] = (calc_eigendecomposition(timeseries[:,cortex_ind], basis)).T    
#%%
#Empirical data FC
triu_indices=np.triu_indices(num_parcels,k=1)
empirical_time=calc_parcellate(data_parcellation,timeseries)
empirical_time=calc_normalize_timeseries(empirical_time.T)
FC_emp = (empirical_time.T@empirical_time)/T
FCvec_emp=FC_emp[triu_indices]

#%%
#Compare reconstructions to data
import matplotlib.pyplot as plt
corr_FC=np.zeros((50,))
corr_FC1=np.zeros((50,))
for mode in range(num_modes):
    if mode==0:
        recon_temp = evec[cortex_ind, 0:mode+1]@np.reshape(np.squeeze(recon_beta[0:mode+1,:,mode]),[1,-1])
        recon_temp1 = evec[cortex_ind, 0:mode+1]@np.reshape(np.squeeze(recon_beta1[0:mode+1,:,mode]),[1,-1])
    else:
        recon_temp = evec[cortex_ind, 0:mode+1]@np.squeeze(recon_beta[0:mode+1,:,mode])
        recon_temp1 = evec[cortex_ind, 0:mode+1]@np.squeeze(recon_beta1[0:mode+1,:,mode])
    final_recon_temp=np.zeros((T,num_vertex))
    final_recon_temp[:,cortex_ind]=recon_temp.T
    
    final_recon_temp1=np.zeros((T,num_vertex))
    final_recon_temp1[:,cortex_ind]=recon_temp1.T
    
    reconstruct_time=calc_parcellate(data_parcellation, final_recon_temp)
    reconstruct_time=calc_normalize_timeseries(reconstruct_time.T)
    
    reconstruct_time1=calc_parcellate(data_parcellation, final_recon_temp1)
    reconstruct_time1=calc_normalize_timeseries(reconstruct_time1.T)
    
    # 
    FC_recon=(reconstruct_time.T@reconstruct_time)/T
    FCvec_recon=FC_recon[triu_indices]
    
    FC_recon1=(reconstruct_time1.T@reconstruct_time1)/T
    FCvec_recon1=FC_recon1[triu_indices]
    
    corr_FC[mode]=np.corrcoef(FCvec_emp,FCvec_recon)[0,1]
    corr_FC1[mode]=np.corrcoef(FCvec_emp,FCvec_recon1)[0,1]

    fig,ax=plt.subplots(1,3,figsize=(8,4))
    ax[0].imshow(FC_emp/np.max(FC_emp),cmap='seismic',vmin=-1,vmax=1)
    im=ax[1].imshow(FC_recon/np.max(FC_recon),cmap='seismic',vmin=-1,vmax=1)
    ax[2].imshow(FC_recon1/np.max(FC_recon1),cmap='seismic',vmin=-1,vmax=1)
    fig.colorbar(im,ax=ax,shrink=0.5)
    fig.suptitle('%d modes'%(mode+1))
    fig.savefig('reconstruction_EDR_%d.png'%(mode+1),dpi=300,bbox_inches='tight')

plt.figure(figsize=(4,4))
plt.plot(np.arange(1,51),corr_FC)
plt.plot(np.arange(1,51),corr_FC1)
plt.xlabel('# eigenmodes')
plt.ylabel('Pearson corr. FCs' )
plt.legend(['Surface','A'])
plt.savefig('pearson_correlation_FC_EDR.png',dpi=300,bbox_inches='tight')