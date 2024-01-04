#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:22:23 2023

@author: Floris Hermanns
"""
from numpy.random import seed
from pathlib import Path

#from sklearn.decomposition import KernelPCA
from fmch.dimred_funs import dimred_hsi
seed(10005)

wdirexp = Path(__file__)

# AE history: ncomp=4_encoder=[20], ncomp=10_encoder=[30], ncomp=20_encoder=[40] & nepoch=40
ae_params = {'name':'AE', 'loss':'SID', 'nepoch':50, 'encoder':[30], 'wdl':0.0001}
net, dataZ_ae = dimred_hsi(wdir=wdirexp, hsi_file='DR_hsi_transp_PRISMA_bg_ref.h5',
                           ncomp=10, mparams=ae_params, plot=False, save=True)

sivm_params = {'name':'SiVM', 'dist_measure':'l2'} # 10 & 20 still to be done!
lvm, dataZ_sivm = dimred_hsi(wdir=wdirexp, hsi_file='DR_hsi_transp_PRISMA_bg_ref.h5',
                             ncomp=4, mparams=sivm_params, plot=True)

pca_params = {'name':'PCA'}
pca, dataZ_pca = dimred_hsi(wdir=wdirexp, hsi_file='DR_hsi_transp_PRISMA_bg_ref.h5',
                            ncomp=4, mparams=pca_params, plot=True)
