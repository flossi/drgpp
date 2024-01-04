#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:25:28 2023

@author: flossi
"""
import numpy as np
import pandas as pd
import rasterio as rio
import matplotlib.pyplot as plt
import h5py

import tensorflow as tf
from deephyp import data
from deephyp import autoencoder
from pymf.sivm import SIVM
from sklearn.decomposition import PCA

from random import sample as rsample
import warnings


def fnv(values, target):
    '''
    A convenience function that finds the index of a value in a list closest
    to a target value. Can be used to select certain wavelengths of remote
    sensing sensors.
    '''
    if target > max(values) + 3:
        warnings.warning(f'Max wavelength is {max(values)} and target is {target}.'
                       ' Will proceed with max WL.')
    if target < min(values) - 3:
        warnings.warning(f'Min wavelength is {min(values)} and target is {target}.'
                       ' Will proceed with min WL.')
    if type(values) == list:
        idx = min(range(len(values)), key=lambda i:abs(values[i]-target))
    elif type(values) == np.ndarray:
        idx = np.abs(values-target).argmin()
    else:
        raise ValueError('Wavelength values should be provided as list or np.ndarray')
    return idx

def dimred_hsi(wdir, hsi_file, ncomp, mparams, plot = False, save = False):
    '''
    Perform dimension reduction on hyperspectral data with different methods
    Args:
        wdir (pathlib.PosixPath): The user directory.
        hsi_file (string): File name of the 2D transformed concatenated HSI.
        ncomp (int): Number of dimensions input data should be reduced to.
        mparams (dict): Names and hyperparameters of the different methods.
            Note that some AE parameters like network architecture and opti-
            misation method are hardcoded. General parameters:
            'name' ('AE' | 'SiVM' | 'kPCA'): autoencoder, simplex volume max-
                imisation or kernel principal component analysis.
            AE specific parameters:
            'loss' ('SSE' | 'SID' | 'CSA' | 'SA'): Loss function to be applied.
                With deephyp, can be sum of squared errors, spectral information
                divergence, cosine spectral angle or spectral angle.
            'nepoch' (int): Number of iterations.
            'encoder' (list of ints): Length of list is the number of layers in
                the encoder. Values are the number of neurons in each layer
                (excluding the code layer).
            'wdl' (float): Weight decay lambda, a regularisation parameter.
            SiVM specific parameters:
            'dist_measure' ('l2' | 'cosine' | 'l1' | 'kl'): SiVM only. 'l2'
                maximises the volume of the simplex.
            kPCA specific parameters:
            'kernel' ('linear' | 'poly' | 'rbf' | 'sigmoid' | 'cosine'): Kernel
                used for PCA.
            'gamma' (float): Kernel coefficient (for rbf, poly and sigmoid
                kernels). sklearn default = None.
            'degree' (int):  Degree for poly kernels. sklearn default = 3.
            'coef0' (float): Independent term (for poly and sigmoid kernels).
                sklearn default = 1.
        plot (bool, optional): If true, scatterplots of the resulting latent
            components are saved in the model directory.
        save (bool, optional): If true, base and coefficient matrices are saved.

    '''
    data_dir = wdir / 'data/dr_files'
    print(f'Reading from {data_dir}')
    model_dir = wdir / f'data/dr_files/ae_models/{mparams["name"]}/{mparams["name"]}{ncomp:02d}'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(wdir / 'out' / 'hsicos_dr' / hsi_file, mode='r') as f:
        spectra = f['ds1'][:, :66] # VNIR only
    spectra[spectra == 0] = .00001 # AE can't deal with zeros
    
    # Load wavelengths from example file
    with rio.open(wdir / 'data/PRISMA/PRS_L2D_STD_20221018105751_FR-Aur_6km_crop.tif') as src:
        wls = [float(w) for w in list(src.descriptions)]
    wls_vnir = [round(x) for x in wls[:66]]
    colnames = [f'Comp{str(x).zfill(2)}' for x in range(1, ncomp + 1)]
    
    val_sample = rsample(range(0, len(spectra)), 10000)
    ival_sample = np.setdiff1d(np.arange(len(spectra)), val_sample)

    if mparams['name'] == 'AE':
        datatrain = data.Iterator(dataSamples=spectra[ival_sample, :], targets=spectra[ival_sample, :], batchSize=10000)
        dataval = data.Iterator(dataSamples=spectra[val_sample, :], targets=spectra[val_sample, :])
        datatrain.shuffle()
    else:
        shuffle_ix = np.random.permutation(np.shape(spectra)[0])
        shuffle_ix_inv = np.argsort(shuffle_ix)
        spectra = spectra[shuffle_ix]
    
    if mparams['name'] == 'AE':
        plot_dir = model_dir / f'Plots_MLPsimple_{mparams["loss"]}_zdim{ncomp:02d}_nepoch{mparams["nepoch"]:03d}_wdl{mparams["wdl"]}'
        h5_file = f'DR_{mparams["name"]}{ncomp:02d}_{mparams["nepoch"]:03d}epoch_PRISMA_bg_ref.h5'
        title = f'model:MLP - loss:{mparams["loss"]} - epochs:{mparams["nepoch"]} - wd_lambda:{mparams["wdl"]}'
        if mparams['loss'] == 'SID':
            acti = ['sigmoid', 'sigmoid']
        else:
            acti = ['relu', 'linear']
    
        mod = autoencoder.mlp_1D_network(inputSize=spectra.shape[1],
                                         encoderSize=mparams['encoder'] + [ncomp], # similar to reduction in PhD thesis: file:///home/hermanns/Downloads/windrim_l_thesis.pdf (ch. 4.3.1)
                                         activationFunc=acti[0],
                                         weightInitOpt='gaussian',
                                         tiedWeights=[1,0],
                                         skipConnect=False,
                                         activationFuncFinal=acti[1])
        mod.add_train_op(name='prisma', lossFunc=mparams['loss'],
                         learning_rate=0.001, method='Adam', wd_lambda=mparams['wdl'])
        datatrain.reset_batch()
        
        mod.train(dataTrain=datatrain, dataVal=dataval, train_op_name='prisma',
                  n_epochs=mparams['nepoch'], save_addr=model_dir, save_epochs=[mparams['nepoch']])
        mod.add_model(addr=model_dir / f'epoch_{mparams["nepoch"]}', modelName='prisma_mlp')
        dataZ = mod.encoder(modelName='prisma_mlp', dataSamples=spectra)
        
        # Export weight, bias and activation (of 1st hidden layer) arrays
        if save:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.restore(sess, str(model_dir / f'epoch_{mparams["nepoch"]}' / 'model.ckpt'))
    
                ae_a1 = mod.a['a1'].eval(feed_dict={mod.x: spectra}, session=sess)
                ae_w1 = mod.weights['encoder_w1'].eval(session=sess)
                ae_w2 = mod.weights['encoder_w2'].eval(session=sess)
                ae_b1 = mod.biases['encoder_b1'].eval(session=sess)
            ae_w1 = pd.DataFrame(ae_w1, index=wls_vnir, columns=[f'Neuron{str(x).zfill(2)}' for x in range(1, mparams['encoder'][0] + 1)])
            ae_w1 = ae_w1.append(pd.Series(ae_b1, name='bias', index=[f'Neuron{str(x).zfill(2)}' for x in range(1, mparams['encoder'][0] + 1)]))
            ae_w2 = pd.DataFrame(ae_w2, index=[f'Neuron{str(x).zfill(2)}' for x in range(1, mparams['encoder'][0] + 1)], columns=colnames)
            
            h5f = h5py.File(data_dir / f'{h5_file[:-3]}_acti1.h5', 'w')
            h5f.create_dataset('acti', data=ae_a1)
            h5f.close()
            ae_w1.to_csv(data_dir / f'{h5_file[:-9]}w1+bias_matrix.csv', index=True)
            ae_w2.to_csv(data_dir / f'{h5_file[:-9]}w2_matrix.csv', index=True)
        
    elif mparams['name'] == 'SiVM':
        plot_dir = model_dir / f'Plots_SiVM{ncomp:02d}_distm-{mparams["dist_measure"]}'
        h5_file = f'DR_{mparams["name"]}{ncomp:02d}_distm-{mparams["dist_measure"]}_PRISMA_bg_ref.h5'
        title = f'model:SiVM - dist_measure:{mparams["dist_measure"]}'
        
        mod = SIVM(spectra.T, num_bases=ncomp, dist_measure=mparams['dist_measure'])
        mod.factorize()
        #os.system('spd-say "matrix factorization has finished"')
        dataZ = mod.H.T
        dataZ = dataZ[shuffle_ix_inv,:] # invert shuffling for LC pixel values
        if save:
            sivm_base = pd.DataFrame(mod.W, index=wls_vnir, columns=colnames)
            sivm_base.to_csv(data_dir / f'{h5_file[:-9]}base_matrix.csv', index=True)
        
    elif mparams['name'] == 'PCA':
        plot_dir = model_dir / f'Plots_PCA{ncomp}'
        title = 'model:PCA'
        h5_file = f'DR_{mparams["name"]}{ncomp:02d}_PRISMA_bg_ref.h5'
        
        mod = PCA(n_components=ncomp)
        dataZ = mod.fit_transform(spectra)
        dataZ = dataZ[shuffle_ix_inv,:]
        if save:
            pca_loads = mod.components_.T * np.sqrt(mod.explained_variance_)
            pca_loads = pd.DataFrame(pca_loads, index=wls_vnir, columns=colnames)
            pca_loads.to_csv(data_dir / f'{h5_file[:-9]}loadings_matrix.csv', index=True)
    if plot == True:
        lcs = [(x, x+1) for x in np.arange(ncomp, step=2)]
        plot_dir.mkdir(parents=True, exist_ok=True)
        for l in lcs:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.hexbin(dataZ[:, l[0]], dataZ[:, l[1]], gridsize=50, cmap='inferno')
            #ax.scatter(dataZ[:, 0], dataZ[:, 1])
            ax.set_xlabel('Latent component {}'.format(l[0]))
            ax.set_ylabel('Latent component {}'.format(l[1]))
            ax.set_title(title)
            fig.savefig(plot_dir / f'LC{l[0]}-{l[1]}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
    if np.isnan(dataZ).any():
        print('WARNING: NA values found in dimension-reduced data matrix!')
    
    if save:
        h5f = h5py.File(data_dir / h5_file, 'w')
        h5f.create_dataset('comps', data=dataZ)
        h5f.close()
    
    return mod, dataZ