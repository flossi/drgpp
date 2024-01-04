#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 04, 2024

@author: Floris Hermanns
"""

import pandas as pd
pd.set_option('display.max_columns', 6)
pd.set_option('display.max_rows', 20)
from pathlib import Path
import datetime as dt

import fiona
fiona.supported_drivers['KML'] = 'rw'

from fmch.hsicos import HSICOS, _build_icos_meta

'''
Folder structure:
Input flux data (e.g. CSV files from ICOS ETC L2 Archives)
must be put in /data/fluxes. Input hyperspectral (PRISMA) data must be put in
/data/PRISMA. The data base of PRISMA imagery used in the analysis is located
in /data/prisma_db.csv.

Output files are generated in /out. The folder already contains output .gpkgs
that are ready for use in statistical analysis with prisma_gpp_stats_main.R.

Python modules are stored in /fmch/fmch and R functions are stored in /R.

'''
#%% Load flux data

wdirexp = Path(__file__)

flx, mask_param_df = _build_icos_meta()
img_db_file = 'prisma_db.csv'

df = '%Y-%m-%d %H:%M:%S'
flx_prisma = pd.read_csv(wdirexp / 'data' / img_db_file, parse_dates=['startdate'],
                        date_format=df, dtype={'dataTakeID': str}).reset_index(drop=True)
flx_prisma['date'] = [dt.datetime.date(x).isoformat() for x in flx_prisma['startdate']]
flx_prisma['icostime'] = (flx_prisma.startdate + pd.Timedelta(hours=1)).dt.round('30min').dt.time

#%% Combining flux and hyperspectral data with HSICOS class

icos_l0 = flx_prisma.name.unique().tolist()

zonal = True
upw = False
odir = 'hsicos_dr'
prisma_gpp = HSICOS(img_csv=img_db_file, wdir=wdirexp, do_mkdir=True, out_dir=odir)

# Import geometry from ICOS L2 data and target (projected) CRS info from imagery
prisma_gpp.crs_and_cropping(icos_l0, zip_path=prisma_gpp.img_dir, overwrite=False, save_csv=False)

# QC overviews (DESIS & PRISMA)
test_qcdf = prisma_gpp.hsi_qc(icos_l0, overwrite=False, save=True)
# After manual inspection of imagery, add a 'usable' column to the csv and enter for each image:
# 0 for unusable, 1 for usable, 2 for maybe usable (requires closer inspection -> FFPs),
# 3 for missing flux data (= later usable)

# Download precip + temp data from E-OBS via Copernicus CDS
prisma_gpp.icos_cds_get(icos_l0, eobs=True) # for FFP & zonal
# Calculate SPEI components from E-OBS data
prisma_gpp.icos_eobs_pet(icos_l0)

###############################################################################
# SPEI computation in R: hsi_gpp_spei_main.R
###############################################################################

# Exlude unusable imagery
icos_l1 = prisma_gpp.update_img_db(img_db_file, era_blh=None, save=True) # zonal -> no era_blh list needed

# Download Copernicus PPI data
prisma_gpp.icos_ppi_get(icos_l1, dataset = 'VI', day_range=10)

### Now either:
## (1) dimension reduction and followup aggregation of DR imagery
# FP modeling & cropping of HSI
flx_geom_gdf, icos_l2 = prisma_gpp.hsi_gdf_prep(icos_l1, 'GPP_DT_VUT_50', upw=upw, zonal=zonal)
'''important note: flx_geom_gdf is shorter than flx_hsi_gdf from (2) as cropping
and therefore removal of HSI NAs takes place after DR'''

# Add SPEI & PPI values to data frame (before DR to remove PPI=0 obs)
flx_geom_gdf2, mask_param_df2 = prisma_gpp.hsi_add_spei_ppi(
    flx_geom_gdf, mask_param_df, dimred=True, zonal=zonal, rm_missing=True,
    rm_sites=['IT-Lsn'], save=True)
#flx_geom_gdf2.loc[flx_geom_gdf2.PPI == 0, :] # Check if zeros are "true" zeros or PPI img is flawed

# Reloading pre dimred data example
flx_geom_gdf2, mask_param_df2 = prisma_gpp.load_pre_dimred_db(zonal=zonal, upw=upw)

# All unusable obs. (NaNs, PPI/GPP/... = 0) were removed before DR
trans_cube = prisma_gpp.hsi_dimred_prep(flx_geom_gdf2, mask_param_df2, save_plot=False)

###############################################################################
# DR processing in separate file (02_hsi_gpp_dimred_main.py) due to deephyp requiring Python 3.7/tensorflow 1.x
# The DR file must be executed in a suitable virtual environment.
###############################################################################

ae_files = ['DR_AE04_050epoch_PRISMA_bg_ref.h5', 'DR_AE10_050epoch_PRISMA_bg_ref.h5', 'DR_AE20_040epoch_PRISMA_bg_ref.h5']
sivm_files = [f'DR_{x}_distm-l2_PRISMA_bg_ref.h5' for x in ['SiVM04', 'SiVM10', 'SiVM20']]
pca_files = [f'DR_{x}_PRISMA_bg_ref.h5' for x in ['PCA04', 'PCA10', 'PCA20']]
dr_files = ae_files + sivm_files + pca_files

for fn in dr_files:
    prisma_gpp.dimred_backtransform(fn, flx_geom_gdf2)

    _, _ = prisma_gpp.dimred_geom_crop(fn, flx_geom_gdf2, mask_param_df2, save=True)

# Reloading dimred data example
flx_comp_gdf = prisma_gpp.load_dimred_db('AE04', zonal=zonal, upw=upw)

prisma_gpp.dimred_backtransform('DR_AE20_040epoch_PRISMA_bg_ref.h5', flx_geom_gdf2)
flx_comp_gdf, _ = prisma_gpp.dimred_geom_crop('DR_AE20_040epoch_PRISMA_bg_ref.h5', flx_geom_gdf2, mask_param_df2, save=True)

## (2) direct aggregation of HSI imagery
# self._mask_px is now standalone and ready to be used outside of hsi_dimred_prep
# requires the usual parameters: mask_param (single row), cube, wls, row (of the flx gdf), loc (flx_loc), ext (out_ext for plots)

flx_hsi_gdf, fi, icos_l2 = prisma_gpp.hsi_geom_crop(
    icos_l1, mask_param_df, 'GPP_DT_VUT_50', sr='vnir', zonal=zonal, upw=upw, save=True)

mask_param_df_cor = mask_param_df[(mask_param_df.name + mask_param_df.dataTakeID)\
                                  .isin(flx_hsi_gdf.name + flx_hsi_gdf.dataTakeID)]

flx_hsi_gdf2, mask_param_df2 = prisma_gpp.hsi_add_spei_ppi(
    flx_hsi_gdf, mask_param_df_cor, dimred=False, zonal=zonal, rm_missing=True,
    rm_sites=['IT-Lsn'], save=True)

# Reloading full data example
flx_hsi_gdf2 = prisma_gpp.load_hsi_db(sr='vnir', zonal=zonal, upw=upw)



