#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 18:34:28 2022

@author: hermanns
"""

__version__ = '0.1'
__author__ = 'Floris Hermanns'
__license__ = 'BSD 3-Clause'
__status__ = 'Prototype'

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

import pyeto
import pyproj as proj
from shapely.geometry import Polygon
import rasterio as rio
import rasterio.plot as riop
import rasterio.mask as riom
import h5py
import spectral.io.envi as envi
from shapely import geometry
from shapely.ops import transform as stransform
from functools import reduce
from fmch import ffp
from fmch.HSI2RGB import HSI2RGB
from cv2 import medianBlur, filter2D
from random import randint

from pathlib import Path
import datetime as dt
from time import sleep
import cdsapi
from hda import Client

import re
import json
import requests
import logging
import warnings
import sys
from zipfile import ZipFile

wdir0 = Path(__file__).parent.parent.parent
logger = logging.getLogger('hsicos')
logger.handlers = []
logger.propagate = False
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
#if not logger.hasHandlers():
f = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler = logging.FileHandler(wdir0 / 'hsicos.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(f)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(f)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
logger.info('Running HSICOS module')

def _build_icos_meta():
    '''
    Function to build a geodataframe containing metadata about a number of ICOS
    sites in different European countries. Includes infos about ecosystem type,
    coordinates, start of records, elevation, etc.
    
    Furthermore, creates a dict of (cloud) masking parameters hardcoded for the
    174 usable PRISMA observations of the warm winter ICOS site selection
    remaining in flx_geom_gdf after self._model_geoms has been executed. The
    parameters are required for the self._mask_px method.
    '''
    flx = pd.DataFrame(
        {'name': ['BE-Bra', 'BE-Dor', 'BE-Lcr', 'BE-Lon', 'BE-Maa', 'BE-Vie', 'CH-Aws', 'CH-Cha', 'CH-Dav', 'CH-Fru', 'CH-Lae', 'CH-Oe2', 'CZ-BK1', 'CZ-KrP',
                  'CZ-Lnz', 'CZ-RAJ', 'CZ-Stn', 'CZ-wet', 'DE-Akm', 'DE-Geb', 'DE-Gri', 'DE-GsB', 'DE-Hai', 'DE-HoH', 'DE-Hod', 'DE-Hte', 'DE-Hzd', 'DE-Kli',
                  'DE-Obe', 'DE-RuR', 'DE-RuS', 'DE-RuW', 'DE-Tha', 'DK-Gds', 'DK-Skj', 'DK-Sor', 'DK-Vng', 'ES-Abr', 'ES-Agu', 'ES-Cnd', 'ES-LJu', 'ES-LM1',
                  'ES-LM2', 'FI-Hyy', 'FI-Let', 'FI-Qvd', 'FI-Sii', 'FR-Aur', 'FR-Bil', 'FR-EM2', 'FR-FBn', 'FR-Fon', 'FR-Gri', 'FR-Hes', 'FR-LGt', 'FR-Lam',
                  'FR-Mej', 'FR-Tou', 'IE-Cra', 'IT-BCi', 'IT-BFt', 'IT-Cp2', 'IT-Lav', 'IT-Lsn', 'IT-MBo', 'IT-Ren', 'IT-SR2', 'IT-Tor', 'NL-Loo', 'RU-Fy2',
                  'RU-Fyo', 'SE-Deg', 'SE-Htm', 'SE-Lnn', 'SE-Nor', 'SE-Ros', 'SE-Svb'],
         'ecosystem': ['MF', 'GRA', 'DBF', 'CRO', 'CSH', 'MF', 'GRA', 'GRA', 'ENF', 'GRA', 'MF', 'CRO', 'ENF', 'CRO', 'MF', 'ENF', 'DBF', 'WET', 'WET', 'CRO',
                       'GRA', 'GRA', 'DBF', 'DBF', 'CRO', 'WET', 'DBF', 'CRO', 'ENF', 'GRA', 'CRO', 'ENF', 'ENF', 'ENF', 'WET', 'DBF', 'CRO', 'SAV', 'OSH', 'WSA',
                       'OSH', 'SAV', 'SAV', 'ENF', 'ENF', 'CRO', 'WET', 'CRO', 'ENF', 'CRO', 'MF', 'DBF', 'CRO', 'DBF', 'WET', 'CRO', 'GRA', 'GRA', 'WET', 'CRO',
                       'DBF', 'EBF', 'ENF', 'OSH', 'GRA', 'ENF', 'ENF', 'GRA', 'ENF', 'ENF', 'ENF', 'WET', 'ENF', 'CRO', 'ENF', 'ENF', 'ENF'],
         'lat': [51.3076, 50.3119, 51.1122, 50.5516, 50.9799, 50.3049, 46.5832, 47.2102, 46.8153, 47.1158, 47.4783, 47.2864, 49.5021, 49.5733, 48.68155, 49.44372,
                 49.03598, 49.0247, 53.8662, 51.0997, 50.95004, 52.02965, 51.07941, 52.08656, 52.0013, 54.2103, 50.96381, 50.89306, 50.7867, 50.6219, 50.86591,
                 50.50493, 50.9626, 56.0737, 55.9127, 55.4859, 56.0375, 38.7018, 36.94, 37.915, 36.9266, 39.9427, 39.9346, 61.8474, 60.6418, 60.2952, 61.8327,
                 43.5497, 44.4937, 49.8721, 43.2408, 48.4764, 48.8442, 48.6741, 47.3229, 43.4964, 48.1184, 43.5729, 53.3231, 40.5237, 45.1978, 41.7043, 45.9562,
                 45.740482, 46.0147, 46.5869, 43.73202, 45.84444, 52.16658, 56.4476, 56.4615, 64.182, 56.0976, 58.3406, 60.0865, 64.1725, 64.2561],
         'lon': [4.5198, 4.9681, 3.8504, 4.7462, 5.6319, 5.9981, 9.79042, 8.4104, 9.85591, 8.5378, 8.3644, 7.7337, 18.5369, 15.0788, 16.94633, 16.69651, 17.9699,
                 14.7704, 13.6834, 10.9146, 13.5126, 11.1048, 10.4521, 11.22235, 11.1777, 12.1761, 13.48978, 13.52238, 13.7213, 6.3041, 6.44715, 6.33096, 13.5651,
                 9.3341, 8.4048, 11.6446, 9.1607, -6.78588, -2.0332, -3.2277, -2.7521, -5.77868, -5.77588, 24.2948, 23.9595, 22.3916, 24.1929, 1.1061, -0.95609,
                 3.02065, 5.6787, 2.7801, 1.9519, 7.06465, 2.2841, 1.2379, -1.79635, 1.3747, -7.6418, 14.9574, 10.742, 12.3573, 11.2813, 12.7503, 11.0458, 11.4337,
                 10.29091, 7.57806, 5.74356, 32.9019, 32.9221, 19.5565, 13.419, 13.1018, 17.4795, 19.738, 19.7745],
         'elev_(m)': [16.0, 247.0, 6.0, 167.0, 87.0, 493.0, 1978.0, 393.0, 1639.0, 982.0, 689.0, 452.0, 875.0, 541.0, 150.0, 625.0, 540.0, 426.0, -1.0, 162.0,
                      385.0, 81.0, 438.0, 193.0, 126.0, 0.2, 395.0, 478.0, 734.0, 515.0, 103.0, 610.0, 385.0, 86.0, 2.0, 40.0, 67.0, np.nan, 197.0, 369.0, 1612.0,
                      np.nan, np.nan, 181.0, 125.0, 11.0, 160.0, 250.0, 39.0, 85.0, 436.0, 103.0, 125.0, 310.0, 153.0, 181.0, 40.0, 158.0, 55.0, 20.0, 23.0, 19.0,
                      1363.0, 1.0, 1550.0, 1735.0, 4.0, 2168.0, 25.0, 276.0, 275.0, 270.0, 115.0, 71.0, 45.0, 158.0, 267.0],
         'MAT_(C)': [9.8, 10.3, 10.6, 10.0, 10.3, 7.8, 2.3, 9.5, 2.8, 7.2, 8.3, 9.8, 6.7, 7.1, 9.3, 7.1, 8.7, 7.7, 8.7, 8.5, 7.8, np.nan, 8.34, 9.1, np.nan, 9.2,
                     7.8, 7.6, 5.5, 7.7, 10.0, 7.5, 8.2, 8.3, 8.7, 9.0, 8.1, 18.3, np.nan, 15.8, 16.0, 16.0, 16.0, 3.5, 4.6, 5.4, 3.5, 13.1, 12.9, 10.8, 13.9, 11.4,
                     11.0, 10.0, 12.3, 13.4, 11.7, 14.1, 9.6, 18.0, 14.5, 15.2, 7.8, 13.0, 5.5, 4.9, 14.2, 2.9, 9.8, 3.9, 3.9, 1.2, 7.4, 6.0, 5.7, 1.8, 1.8],
         'MAP_(mm)': [750.0, 828.0, 861.0, 800.0, 839.0, 1062.0, 918.0, 1136.0, 1062.0, 1651.0, 1100.0, 1155.0, 1316.0, 620.0, 550.0, 681.0, 685.0, 604.0, 558.0,
                      470.0, 901.0, np.nan, 744.0, 563.0, np.nan, 645.0, 901.0, 842.0, 996.0, 1033.0, 700.0, 1250.0, 843.0, 883.0, 809.0, 640.0, 961.0, 400.0, np.nan,
                      474.0, 400.0, 700.0, 700.0, 711.0, 627.0, 650.0, 711.0, 669.0, 960.1, 680.0, 700.0, 679.0, 597.0, 889.0, 700.0, 677.0, 722.0, 544.0, 821.0,
                      600.0, 697.0, 805.0, 1291.0, 1100.0, 1365.0, 971.0, 920.0, 920.0, 786.0, 711.0, 711.0, 523.0, 707.0, 558.0, 544.0, 614.0, 614.0],
         'long_name': ['Brasschaat', 'Dorinne', 'Lochristi', 'Lonzee', 'Maasmechelen', 'Vielsalm', 'Alp Weissenstein', 'Chamau', 'Davos', 'Früebüel', 'Lägeren',
                       'Oensingen 2', 'Bily Kriz forest', 'Kresin u Pacova', 'Lanzhot', 'Rajec', 'Stitna', 'Trebon', 'Anklam', 'Gebesee', 'Grillenburg',
                       'Grosses Bruch', 'Hainich', 'Hohes Holz', 'Hordorf', 'Huetelmoor', 'Hetzdorf', 'Klingenberg', 'Oberbärenburg', 'Rollesbroich',
                       'Selhausen Juelich', 'Wüstebach', 'Tharandt', 'Gludsted Plantage', 'Skjern', 'Soroe', 'Voulundgaard', 'Albuera', 'Aguamarga', 'Conde',
                       'Llano de los Juanes', 'Majadas del Tietar North', 'Majadas del Tietar South', 'Hyytiälä', 'Lettosuo', 'Qvidja', 'Siikaneva', 'Aurade',
                       'Bilos', 'Estrees-Mons A28', 'Font-Blanche', 'Fontainebleau-Barbeau', 'Grignon', 'Hesse', 'La Guette', 'Lamasquere', 'Mejusseaume', 'Toulouse',
                       'Clara Raised Bog', 'Borgo Cioffi', 'Bosco Fontana', 'Castelporziano2', 'Lavarone', 'Lison', 'Monte Bondone', 'Renon', 'San Rossore 2',
                       'Torgnon', 'Loobos', 'Fyodorovskoye dry spruce', 'Fyodorovskoye', 'Degerö', 'Hyltemossa', 'Lanna', 'Norunda', 'Rosinedal-3', 'Svartberget'],
         'icos_l2_avail': [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 2, 3, 1, 0, 0, 0, 0, 0, 0, 1, 3, 0, 1,
                           0, 1, 2, 0, 1, 3, 0, 2, 1, 2, 2, 0, 0, 0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]}
                        #0: no L2 parameters; 1: all L2 meteo. params.; 2: all but PBLH, 3: L2 but parameter unchecked
        )
    flx_gdf = gpd.GeoDataFrame(flx, geometry=gpd.points_from_xy(flx.lon, flx.lat))
    flx_gdf.crs = 'EPSG:4326'
    #flx_gdf = flx_gdf.sort_values(by='name')
    
    param_dict = {'name': ['BE-Vie', 'BE-Vie', 'BE-Vie', 'DE-Hai', 'DE-Hai', 'DE-HoH', 'DE-HoH', 'DE-HoH', 'DE-HoH', 'DE-RuS', 'DE-RuS', 'DE-RuS',#11
                           'FI-Hyy', 'FI-Hyy', 'FI-Hyy', 'FI-Hyy', 'FI-Sii', 'FI-Sii', 'FI-Sii', 'FI-Sii', 'FI-Sii', 'FR-Fon', 'FR-Fon', 'FR-Fon',#23
                           'FR-Fon', 'FR-Fon', 'FR-Fon', 'FR-Fon', 'FR-Fon', 'FR-Fon', 'FR-Fon', 'FR-Fon', 'FR-Fon', 'FR-Fon', 'FR-Fon', 'FR-Fon',#35
                           'IT-Lsn', 'IT-SR2', 'IT-SR2', 'IT-SR2', 'IT-SR2', 'IT-SR2', 'IT-SR2', 'IT-SR2', 'IT-SR2', 'IT-SR2', 'IT-SR2', 'IT-SR2',#47
                           'SE-Deg', 'SE-Deg', 'SE-Deg', 'SE-Deg', 'SE-Deg', 'SE-Deg', 'SE-Svb', 'SE-Svb', 'SE-Svb', 'SE-Svb', 'SE-Svb', 'BE-Bra',#59
                           'BE-Bra', 'BE-Bra', 'DE-Tha', 'DE-Tha', 'DE-Tha', 'DE-Tha', 'DE-Tha', 'DE-Tha', 'DE-Tha', 'DE-Tha', 'DE-Tha', 'DE-Tha',#71
                           'DE-Tha', 'IT-Cp2', 'IT-Cp2', 'IT-Cp2', 'IT-Cp2', 'IT-Cp2', 'IT-Cp2', 'IT-Cp2', 'IT-Cp2', 'IT-Cp2', 'IT-Cp2', 'IT-Cp2',#83
                           'IT-Cp2', 'IT-Cp2', 'IT-Cp2', 'IT-Cp2', 'FR-EM2', 'FR-EM2', 'FR-EM2', 'FR-Tou', 'FR-Tou', 'FR-Tou', 'FR-Tou', 'FR-Tou',#95
                           'FR-Tou', 'FR-Tou', 'FR-Tou', 'FR-Tou', 'FR-Tou', 'FR-Tou', 'FR-Tou', 'FR-Tou', 'FR-Tou', 'FR-Tou', 'FR-Tou', 'FR-Tou',#107
                           'DK-Skj', 'FR-Lam', 'FR-Lam', 'FR-Lam', 'FR-Lam', 'FR-Lam', 'FR-Lam', 'FR-Lam', 'FR-Lam', 'FR-Lam', 'FR-Lam', 'FR-Lam',#119
                           'FR-Lam', 'FR-Lam', 'FR-Lam', 'FR-Lam', 'FR-Lam', 'FR-Lam', 'FR-Lam', 'CH-Cha', 'CH-Lae', 'CZ-BK1', 'CZ-BK1', 'DE-Gri',#131
                           'DE-Gri', 'DE-Gri', 'DE-Gri', 'DE-Gri', 'DE-Gri', 'DE-Gri', 'DE-Gri', 'DE-Gri', 'DE-Gri', 'DE-Hzd', 'DE-Hzd', 'DE-Kli',#143
                           'DE-Kli', 'DE-Kli', 'DE-Kli', 'DE-Kli', 'DE-Kli', 'DE-Kli', 'FR-Aur', 'FR-Aur', 'FR-Aur', 'FR-Aur', 'FR-Aur', 'FR-Aur',#155
                           'FR-Aur', 'FR-Aur', 'FR-FBn', 'FR-FBn', 'FR-Hes', 'FR-Hes', 'FR-Mej', 'FR-Mej', 'IT-BFt', 'IT-BFt', 'IT-Ren', 'IT-Ren',#167
                           'IT-Tor', 'IT-Tor', 'SE-Ros', 'SE-Ros', 'SE-Ros', 'FR-LGt'],
                  'dataTakeID': ['20210227104304', '20200420104916', '20220415104920', '20200419103219', '20220802102909', '20200413102845', '20210524102449', '20210710103509', '20220628102530',
                                 '20200426105222', '20210629104532', '20220531104234', '20200411095234', '20200626100214', '20210904095923', '20221009095918', '20200411095234', '20200626100214',
                                 '20210725095226', '20210904095923', '20221009095918', '20200409105954', '20200623105300', '20200912104911', '20200918105227', '20200809110237', '20210427105924',
                                 '20210815105637', '20210311105009', '20210329105948', '20210907105327', '20211122110338', '20220421105311', '20220211110330', '20220323105332', '20220508104637',
                                 '20210213100428', '20191114102719', '20210425102709', '20210610102029', '20210703101726', '20210801101735', '20210807102057', '20210220102450', '20210905102112',
                                 '20211027101806', '20220202101025', '20220721102430', '20200621101505', '20200910101117', '20200818101444', '20200807102443', '20210727102510', '20210813101842',
                                 '20200621101505', '20210727102510', '20210703101146', '20210813101842', '20220628102202', '20210427105834', '20220323105243', '20220508104548', '20200407102538',
                                 '20200806101148', '20200731100829', '20210621100840', '20210303101242', '20210905101911', '20211009100608', '20220326100558', '20220214101559', '20220720100557',
                                 '20221027101232', '20210517100736', '20210731100126', '20210806100448', '20210702100117', '20210904100503', '20210627101429', '20220225100834', '20220121100201',
                                 '20220116101515', '20220830101453', '20220412100139', '20220213100152', '20220714100507', '20220615100452', '20220824101139', '20210210105016', '20211023104639',
                                 '20220804110257', '20200409110112', '20200415110430', '20200623105420', '20200912105029', '20200705110052', '20210415105415', '20210526110037', '20210729110423',
                                 '20210809105434', '20201121105629', '20211023104821', '20210821110117', '20220509110429', '20220601110116', '20220711105129', '20220919105800', '20221116105759',
                                 '20200420104742', '20200409110112', '20200415110430', '20200623105420', '20200705110052', '20200710104733', '20210612105406', '20210809105434', '20201121105629',
                                 '20210907105445', '20211023104821', '20220228105818', '20220415105119', '20220508104758', '20220601110116', '20220711105129', '20221018105751', '20220919105800',
                                 '20221116105759', '20200507104314', '20200507104310', '20200405095241', '20200822094843', '20200806101148', '20200731100829', '20210621100840', '20210303101242',
                                 '20210905101911', '20211009100608', '20220326100558', '20220214101559', '20220720100557', '20221027101232', '20200806101148', '20200731100829', '20200806101148',
                                 '20200731100829', '20210621100840', '20211009100608', '20220214101559', '20220720100557', '20221027101232', '20200415110430', '20200705110052', '20210612105406',
                                 '20220228105818', '20220624105753', '20220508104758', '20220601110116', '20221018105751', '20200414104750', '20200831104351', '20200721103558', '20200808104548',
                                 '20210223111349', '20201128111516', '20211027101743', '20220407101408', '20220220102034', '20221119101021', '20200709103008', '20210808103707', '20200621101505',
                                 '20200910101117', '20200818101444', '20220919105701'],
                  'T1': [3, 0, 3, 0, 0, 10, 10, 10, 3, 0, 10, 3, 10, 0, 10, 3, 10, 1, 10, 10, 3, 3, 3, 3,#23
                         3, 3, 3, 3, 3, 3, 3, 10, 0, 3, 3, 3, 3, 3, 0, 0, 3, 0, 3, 3, 0, 0, 10, 0,#47
                         3, 5, 0, 5, 3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0,#71
                         0, 3, 0, 3, 0, 3, 0, 1, 1, 0, 1, 0, 1, 0, 0, 10, 0, 1, 1, 1, 3, 1, 1, 1,#95
                         1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,#119
                         3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 1,#143
                         1, 1, 1, 1, 1, 1, 0, 0, 3, 0, 1, 1, 0, 0, 1, 1, 3, 0, 0, 0, 1, 1, 1, 1,#167
                         0, 0, 1, 0, 0, 1],
                  'cm+csm': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,#23
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,#47
                             1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0,#71
                             1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,#95
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1,#119
                             1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,#143
                             1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,#167
                             1, 1, 1, 1, 1, 1],
                  'sh_thresh': [.01, .01, .025, .02, .02, .02, .02, .02, .015, .01, .02, .02, .025, .02, .02, .02, .025, .02, .025, .025, .02, .01, .01, .01,#23
                                .01, .01, .01, .02, .025, .02, .02, .03, .01, .02, .01, .035, .02, .015, .01, .01, .02, .01, .02, .017, .01, .02, .02, .01,#47
                                .02, .01, .01, .02, .018, .018, .02, .016, .02, .02, .028, .015, .01, .01, .012, .01, .01, .01, .01, .015, .013, .015, .022, .01,#71
                                .022, .01, .01, .02, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .015, .015, .015, .01, .01, .015, .01,#95
                                .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01,#119
                                .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .013, .01, .01, .01, .02, .02, .02, .02, .02, .02, .02, .02, .02, .02,#143
                                .02, .02, .02, .02, .02, .03, .01, .01, .01, .01, .025, .02, .01, .01, .02, .02, .02, .02, .02, .02, .01, .01, .02, .02,#167
                                .02, .02, .03, .024, .025, .02],
                  'ndvi_thresh': [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5,#23
                                  .5, .5, .5, .5, .5, .5, .5, .5, .6, .5, .5, .5, .5, .5, .6, .55, .5, .5, .5, .5, .5, .5, .5, .5,#47
                                  .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .55, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5,#71
                                  .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .55, .55, .55, .55, .55,#95
                                  .55, .55, .55, .55, .55, .55, .55, .55, .55, .55, .55, .55, .55, .55, .55, .55, .55, .55, .55, .55, .55, .55, .55, .55,#119
                                  .55, .55, .55, .55, .55, .55, .55, .5, .55, .5, .5, .5, .5, .5, .5, .5, .55, .5, .55, .5, .5, .5, .5, .55,#143
                                  .55, .55, .55, .55, .55, .55, .5, .5, .5, .5, .5, .5, .5, .5, .55, .55, .55, .55, .55, .55, .5, .5, .5, .5,#167
                                  .5, .5, .55, .6, .55, .5],
                  'blue_cor': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,#39
                               1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,#79
                               1, 1, 1, 1, 1, 1, .03, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,#119
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,#159
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    mask_p_df = pd.DataFrame.from_dict(param_dict) # is not sorted
    
    return flx_gdf, mask_p_df

def nearest(items, target):
    return min(items, key=lambda x: abs(x - target))

def _fnv(values, target):
    '''
    A convenience function that finds the index of a value in a list closest
    to a target value. Can be used to select certain wavelengths of remote
    sensing sensors.
    '''
    if target > max(values) + 3:
        logger.warning(f'Max wavelength is {max(values)} and target is {target}.'
                       ' Will proceed with max WL.')
    if target < min(values) - 3:
        logger.warning(f'Min wavelength is {min(values)} and target is {target}.'
                       ' Will proceed with min WL.')
    if type(values) == list:
        idx = min(range(len(values)), key=lambda i:abs(values[i]-target))
    elif type(values) == np.ndarray:
        idx = np.abs(values-target).argmin()
    else:
        raise ValueError('Wavelength values should be provided as list or np.ndarray')
    return idx

def _local_mask(raster, transform, shapes, **kwargs):
    '''
    Wrapper for rasterio.mask.mask to allow for in-memory processing.
    From: https://gis.stackexchange.com/questions/329434/creating-an-in-memory-rasterio-dataset-from-numpy-array
    Args:
        raster (numpy.ndarray): raster to be masked with dim: [H, W]
        count (int): number of bands
        transform (affine.Affine): the transform of the raster
        shapes, **kwargs: passed to rasterio.mask.mask
    '''
    with rio.io.MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=raster.shape[0],
            width=raster.shape[1],
            count=raster.shape[2],
            dtype=raster.dtype,
            transform=transform,
        ) as dataset:
            for i in range(raster.shape[2]):
                dataset.write(raster[:,:,i], i+1)
        with memfile.open() as dataset:
            output, out_trans = riom.mask(dataset, shapes, **kwargs)
        if output.ndim == 3:
            output = np.einsum('kli->lik', output)
            
    return output, out_trans

def desis_crop(path, mask, indexes = None):
    '''
    Reads DESIS GeoTiffs and crops using rasterio.mask methods (argument
    all_touched is always set to True).
        
    Args:
        path (string): The file path of the zipped PRISMA image to be cropped.
        mask (GeoJSON-like dict): passed to rasterio.mask.mask. Should have
            ETRS89-extended / LAEA Europe coordinates (epsg:3035).
        indexes (int / list of ints, optional): Indexes of the bands to be
            included. Use Python indexing starting at 0.
    '''
    if indexes == None:
        indexes = list(range(0, 235)) # all DESIS bands by default
    if isinstance(indexes, int):
        indexes_rio = indexes + 1
    elif isinstance(indexes, list):
        indexes_rio = [x + 1 for x in indexes]
    hdr = envi.read_envi_header(path.parent / (path.stem + '.hdr'))
    wls = [float(i) for i in hdr['wavelength']]
    with rio.open(path, driver='GTiff') as src:
        epsg = src.crs.to_epsg()
    # (re-)project mask derived from ICOS coordinates to HSI CRS
    crs_lam = proj.CRS.from_epsg('3035')
    crs_utm = proj.CRS.from_epsg(epsg)
    transf = proj.Transformer.from_crs(crs_lam, crs_utm, always_xy=True)
    utm_mask = stransform(transf.transform, mask)
    with rio.open(path, driver='GTiff') as src:
        raw_cube, out_trans = riom.mask(src, shapes=[utm_mask], all_touched=True,
                                        crop=True, indexes=indexes_rio)
    if raw_cube.ndim == 2:
        out_ext = riop.plotting_extent(raw_cube, out_trans)
        temp = raw_cube.T
        gain = float(hdr['data gain values'][indexes])
        offset = float(hdr['data offset values'][indexes])
        na_val = float(hdr['data ignore value']) * gain
    else:
        out_ext = riop.plotting_extent(raw_cube[0], out_trans)
        temp = np.einsum('kli->lik', raw_cube)
        gain = np.array([float(x) for x in hdr['data gain values']])\
            [np.newaxis, np.newaxis, indexes]
        offset = np.array([float(x) for x in hdr['data offset values']])\
            [np.newaxis, np.newaxis, indexes]
        na_val = (float(hdr['data ignore value']) * gain[:,:,0]).item()
    ff_cube = offset + gain * temp # L=OffsetOfBand+GainOfBand*DN
    ff_cube = ff_cube.astype('float32')
    
    return ff_cube, wls, out_ext, na_val, out_trans, epsg

def prisma_crop(path, mask, indexes = None, swir = False):
    '''
    Reads zipped PRISMA imagery and crops using rasterio.mask methods (argument
    all_touched is always set to True).
    
    Args:
        path (string): The file path of the zipped PRISMA image to be cropped.
        mask (GeoJSON-like dict): passed to rasterio.mask.mask. Should have
            ETRS89-extended / LAEA Europe coordinates (epsg:3035).
        indexes (int / list of ints, optional): Indexes of the bands to be
            included. Use Python indexing starting at 0.
        swir (bool, optional): If true, bands in the 1000-2500 nm range are
            imported and concatenated with the VNIR cube.
    '''
    with ZipFile(path) as zf:
        for file in zf.namelist():
            if not file.endswith('.he5'): # optional filtering by filetype
                logger.error('{}: Zip doesnt contain .he5 file.'.format(path))
                continue
            with zf.open(file) as f:
                with h5py.File(f, mode='r') as h5f:
                    vnir_wls = pd.Series(
                        np.flip(h5f.attrs['List_Cw_Vnir'])).drop(range(63, 66))
                    vnir_raw = h5f['/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube'][:]
                    #err_mat = h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_PIXEL_L2_ERR_MATRIX'][:]
                    
                    geo = {'xmin': min(h5f.attrs['Product_ULcorner_easting'], h5f.attrs['Product_LLcorner_easting']),
                           'xmax': max(h5f.attrs['Product_LRcorner_easting'], h5f.attrs['Product_URcorner_easting']),
                           'ymin': min(h5f.attrs['Product_LLcorner_northing'], h5f.attrs['Product_LRcorner_northing']),
                           'ymax': max(h5f.attrs['Product_ULcorner_northing'], h5f.attrs['Product_URcorner_northing']),
                           'proj_code': h5f.attrs['Projection_Id'],
                           'proj_name': h5f.attrs['Projection_Name'],
                           'proj_epsg': h5f.attrs['Epsg_Code']
                           }
                    smax = h5f.attrs['L2ScaleVnirMax']
                    smin = h5f.attrs['L2ScaleVnirMin']
                    if swir == True:
                        swir_wls = pd.Series(
                            np.flip(h5f.attrs['List_Cw_Swir'])).drop(range(0, 6))
                        swir_raw = h5f['/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube'][:]
                        smax_sw = h5f.attrs['L2ScaleSwirMax']
                        smin_sw = h5f.attrs['L2ScaleSwirMin']
                    
    rows0,_,cols0 = np.shape(vnir_raw)
    geo['xsize'] = (geo['xmax']-geo['xmin'])/cols0
    geo['ysize'] = (geo['ymax']-geo['ymin'])/rows0
    
    bbox = [geo['xmin'], geo['ymin'], geo['xmax'], geo['ymax']]
    
    vnir_cube = np.einsum('kli->kil', np.flip(vnir_raw, 1))
    vnir_refl = (smin + vnir_cube * (smax - smin)) / 65535
    
    if swir == True:
        swir_cube = np.einsum('kli->kil', np.flip(swir_raw, 1))
        swir_refl = (smin_sw + swir_cube * (smax_sw - smin_sw)) / 65535
        
        full_cube = np.concatenate([vnir_refl[:, :, vnir_wls.index],
                                    swir_refl[:, :, swir_wls.index]], axis=2)
        wls = pd.concat([vnir_wls, swir_wls]).reset_index(drop=True)
    else:
        full_cube = vnir_refl[:, :, vnir_wls.index]
        wls = vnir_wls
    
    if indexes == None:
        indexes_rio = [x + 1 for x in wls.index.tolist()]
    elif isinstance(indexes, int):
        indexes_rio = indexes + 1
    elif isinstance(indexes, list):
        indexes_rio = [x + 1 for x in indexes]

    rows, cols = np.shape(full_cube)[0:2]

    src_trans = rio.transform.from_bounds(*bbox, width=cols, height=rows)
    
    # (re-)project mask derived from ICOS coordinates to HSI CRS
    epsg = geo['proj_epsg']
    crs_lam = proj.CRS.from_epsg('3035')
    crs_utm = proj.CRS.from_epsg(epsg)
    transf = proj.Transformer.from_crs(crs_lam, crs_utm, always_xy=True)
    utm_mask = stransform(transf.transform, mask)
    # In rio.Affine: last value is ymax, not ymin!
    ff_cube, out_trans = _local_mask(full_cube, src_trans, [utm_mask], crop=True,
                                    all_touched=True, indexes=indexes_rio)
    out_ext = riop.plotting_extent(ff_cube, out_trans)
    na_val = 0
    
    return ff_cube, wls, out_ext, na_val, out_trans, epsg


class HSICOS():
    
    def __init__(self, img_csv, wdir = wdir0, do_mkdir = False, out_dir = False, era_blh = []):
        '''
        Args:
            wdir (string): The directory where BSQ files and the GeoPackage
                with ROI bounding box are located.
            img_csv (string): Name of the metadata CSV file that contains key
                info about the imagery to be processed. Required columns for
                cropping are:
                'name': Name or abbr. of the corresp. ICOS site
                'startdate': Timestamp of image acquisition (YYYY-MM-DD hh:mm:ss)
                'dataTakeID': datatake ID of the scene
                'clouds': commentary field to mark missing data, etc.
                'era_pblh': planetary boundary layer height value from ERA5 re-
                    analysis. Only used when PBLH is missing from ICOS L2 data.
                Colums 'date' and 'icostime' are created when the class is ini-
                tiated to match ICOS standards (e.g. conversion of TZ from UTC
                to CET)
            do_mkdir (bool, optional): If true, the output directory is created.
            out_dir (string, optional): Set a custom name for the working
                folder, e.g. to continue previously produced results. If
                unused, a generic output directory name will be used.
        '''
        self._wdir = Path(wdir)
        self.icos_dir = self._wdir / 'data' / 'fluxes'
        self._df_icos = '%Y%m%d%H%M'
        self._df = '%Y-%m-%d %H:%M:%S'
        
        self.img_db = pd.read_csv(
            self.icos_dir.parent / img_csv, parse_dates=['startdate'],
            date_format=self._df, dtype={'dataTakeID': str}).reset_index(drop=True)
        
        # check if metadata table is suitable
        if all(x in self.img_db.columns for x in \
               ['name', 'startdate', 'dataTakeID', 'clouds']):
            pass
        else:
            raise ValueError('Missing columns in "img_db". Please prepare the '\
                             'data frame according to documentation.')
        # check datetime format
        checkdate = self.img_db[['name', 'startdate']].select_dtypes(include=[np.datetime64])
        if checkdate.shape[1] == 0:
            raise ValueError('Incorrect datetime format, should be np.datetime64.')
        
        self.img_db['date'] = [dt.datetime.date(x).isoformat() for x in self.img_db['startdate']]
        self.img_db['icostime'] = (self.img_db.startdate +
                                   pd.Timedelta(hours=1)).dt.round('30min').dt.time
        self.img_db['wls'] = 0
        self.img_db['wls'] = self.img_db['wls'].astype('object')

        self.icos_db, _ = _build_icos_meta()
        
        # get sensor type from ID formatting
        if '_' in self.img_db.dataTakeID.iloc[0]:
            self.sensor = 'DESIS'
        else:
            self.sensor = 'PRISMA'
        self.img_dir = self._wdir / 'data' / self.sensor
        
        # create output folder
        ssm = round((dt.datetime.now() - dt.datetime.now().replace(hour=0, minute=0,second=0, microsecond=0)).total_seconds())
        if out_dir == False:
            fnamestr = 'hsicos_out_' + dt.date.today().isoformat() + str(ssm)
        elif isinstance(out_dir, str):
            fnamestr = out_dir
        else:
            raise ValueError('Fname must be a string.')
        self.out_dir = self._wdir / 'out' / fnamestr
        if do_mkdir == True:
            try:
                self.out_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                logger.info('Output directory already exists')
        
        # Aggregated geom info for ICOS sites is loaded if available
        flx_loc_path = self.out_dir / f'{self.sensor}_hsicos_crs_loc.csv'
        if flx_loc_path.exists():
            flx_df = pd.read_csv(flx_loc_path, on_bad_lines='skip')
            self.flx_loc = gpd.GeoDataFrame(flx_df, geometry=gpd.points_from_xy(
                flx_df.lon, flx_df.lat), crs='EPSG:4326')
            logger.info('CRS and location overview of ICOS sites loaded to self.flx_loc.')
        else:
            logger.info('CRS and location overview of ICOS sites could not be loaded. Create with crs_and_cropping method.')
        
        # check existing cropped PPI files
        self.img_db['ppi_file'] = ''
        ppi_dir = self.img_dir.parent / 'Copernicus/S2PPI'
        ppif_list = [x.name for x in list(ppi_dir.glob('*VI_PPI*.tif'))]
        
        for fname in ppif_list:
            self.img_db.loc[(self.img_db.name == fname[-25:-19]) &
                            (self.img_db.dataTakeID == fname[-18:-4]), 'ppi_file'] = fname
        logger.info('S2 PPI images locally available for {} data takes.'\
                    .format(len(self.img_db.loc[self.img_db.ppi_file != '', :])))
        

        
    
### QUALITY CHECKS ############################################################

    def crs_and_cropping(self, icos_list, zip_path = None, date = None,
                        overwrite = False, save_csv = False):
        '''
        Imports geospatial information for ICOS sites into the class and crops
        HSI to a 3km buffer around the site. Location is imported from ICOS
        data in WGS84. Projected CRS is imported from the hyperspectral data
        inside the PRISMA_crop function. It is assumed that all imagery for a
        single ICOS site has the same CRS (which might not be the case in very
        rare cases). Crops are saved as GeoTIFF in self.img_dir. 
        
        Args:
            icos_list (string / list of strings): Abbreviation of the ICOS
                site(s) for which geometry information will be collected.
            zip_path (string, optional): Location of zipped PRISMA imagery,
                e.g. if located on external drive instead of self.img_dir.
            date (string, optional): Date of image to be cropped. If no date is
                supplied, all available dates for the site will be processed.
            overwrite (bool, optional): If true, imagery for which a cropped
                file already exists will not be skipped.
            save_csv (bool, optional): If true, the collected CRS & geographic
                information will be saved inside the imagery folder.

        '''
        if isinstance(icos_list, str):
            icos_list = [icos_list]
        
        if zip_path:
            img_dir = Path(zip_path)
        else:
            img_dir = self.img_dir
        
        crslist = [0]*len(icos_list)
        lons = [0]*len(icos_list)
        lats = [0]*len(icos_list)
        for j, site in enumerate(icos_list):
            if date:
                datelist = self.img_db.loc[(self.img_db.name == site) &
                                           (self.img_db.date == date), 'startdate'].dt.date
            else:
                datelist = self.img_db.loc[self.img_db.name == site, 'startdate'].dt.date
            dtakes = self.img_db.loc[datelist.index, 'dataTakeID']
            
            if self.sensor == 'DESIS':
                img_paths = [list((img_dir).glob('DESIS-HSI-L2A-DT{}*SPECTRAL*.tif'\
                                                 .format(dt))) for dt in dtakes]
                crop_paths = [self.img_dir / 'DESIS-HSI-L2A-DT{}_{}_6km_crop.tif'\
                              .format(dt.zfill(14), site) for dt in dtakes]
            elif self.sensor == 'PRISMA':
                img_paths = [list((img_dir).glob('PRS_L2D_STD_{}*.zip'\
                                                 .format(dt))) for dt in dtakes]
                crop_paths = [self.img_dir / 'PRS_L2D_STD_{}_{}_6km_crop.tif'\
                              .format(dt, site) for dt in dtakes]
                
            crop_exists = [p.is_file() for p in crop_paths]
            crs_import_path = [p for p in crop_paths if p.is_file()]
            no_crop = [i for (i, v) in zip(dtakes, crop_exists) if not v]
            if len(no_crop) != 0:
                logger.info('{}: No cropped image found for data takes {}.'.format(site, no_crop))
            
            # 1) Load flux locations
            epsg_tower_loc = self._crs_calc(site)
            flx_loc = self.icos_db.loc[self.icos_db.name == site, 'geometry'].to_crs(3035)
            lons[j] = self.icos_db.loc[self.icos_db.name == site, 'geometry'].item().x
            lats[j] = self.icos_db.loc[self.icos_db.name == site, 'geometry'].item().y
            flx_roi_crop = flx_loc.buffer(3000) # lambert -> buffer in meters
            box_geom_crop = geometry.box(*flx_roi_crop.total_bounds)
            
            if all(crop_exists):
                logger.warning('{}: All HSI are already cropped.'.format(site))
                if overwrite == False:
                    try:
                        with rio.open(crs_import_path[0]) as src:
                            crslist[j] = src.crs.to_epsg()
                    except IndexError:
                        logger.warning('{}: No CRS could be imported. Deriving UTM CRS from longitudes.'.format(site))
                        crslist[j] = epsg_tower_loc
                    continue
            # 2) Load CRS & crop
            cubes = [0]*len(img_paths)
            wlss = [0]*len(img_paths)
            blength = []
            img_crss = [0]*len(img_paths)
            itransforms = [0]*len(img_paths)
            for i, path in enumerate(img_paths):
                if crop_exists[i]: # only process paths for which cropped imagery is missing:
                    logger.info('{}: Image DT:{} has already been cropped.'\
                                .format(site, dtakes.iloc[i]))
                    if overwrite == False:
                        continue
                if len(path) == 0:
                    logger.info('{}: Image DT:{} could not be found. Skipping.'\
                                .format(site, dtakes.iloc[i]))
                    continue
                elif len(path) > 1:
                    raise ValueError('{}: DT:{} not unique. Please investigate.'\
                                     .format(site, dtakes.iloc[i]))
                if self.sensor == 'DESIS': # cubes = 6km crops for saving subset as GeoTIFF
                    cubes[i], wlss[i], ext, na_val, itransforms[i], img_crss[i] =\
                        desis_crop(path[0], box_geom_crop)
                elif self.sensor == 'PRISMA':
                    cubes[i], wlss[i], ext, na_val, itransforms[i], img_crss[i] =\
                        prisma_crop(path[0], box_geom_crop, swir=True) # SWIR should always be saved in cropped TIFF
                cubes[i] = np.round(cubes[i], 6)
                blength.append(len(wlss[i])) #usually for DESIS 235, for PRISMA 63 (VNIR) / 230 (VSWIR).
            if len(set(blength)) > 1:
                raise ValueError('{}: Wavelengths are not identical for all images. '.format(site) +
                                 'Wavelengths by order in img_db: {}'.format(blength))

            img_crss = [x for x in img_crss if x != 0] # remove zeros from skipped images
            if len(img_crss) == 0:
                logger.info('{}: No HSI to be processed! Skipping site.'.format(site))
                try:
                    with rio.open(crs_import_path[0]) as src:
                        crslist[j] = src.crs.to_epsg()
                except IndexError:
                    logger.warning('{}: No CRS could be imported. Deriving UTM CRS from longitudes.'.format(site))
                    crslist[j] = epsg_tower_loc
                continue
            elif len(set(img_crss)) > 1:
                logger.error('{}: CRS is not identical for all images!'.format(site))
            # unique CRS for all images (usually CRS will be unique across images from a single ICOS site)
            crs_tower_loc = proj.CRS.from_epsg(epsg_tower_loc)
            epsg = pd.Series(img_crss).value_counts().idxmax()
            crs_utm = proj.CRS.from_epsg(epsg)
            if epsg != epsg_tower_loc:
                logger.warning('{}: HSI UTM zone differs from tower location UTM zone. Raster will be reprojected to the latter.'.format(site))
                crslist[j] = crs_tower_loc.to_epsg()
            else:
                crslist[j] = crs_utm.to_epsg()
            for i, cube in enumerate(cubes):
                if not isinstance(cube, np.ndarray): # only process cubes that could be loaded in first inner loop
                    continue
                rows, cols, b = np.shape(cube) # dimension order will be changed back to [b, y, x] when reading with rasterio
                ometa = {'driver': 'GTiff',
                         'dtype': 'float32',
                         'interleave': 'band',
                         'nodata': na_val,
                         'width': cols,
                         'height': rows,
                         'count': b,
                         'crs': crs_utm,
                         'transform': itransforms[i]}
                logger.info('saving crop of data take {} with CRS - EPSG:{}.'\
                            .format(dtakes.iloc[i], epsg))
                wls_string = [str(w) for w in wlss[i]]
                if epsg == epsg_tower_loc:
                    with rio.open(crop_paths[i], 'w', **ometa) as dst:
                        for k in range(0, b):
                            dst.write_band(k+1, cube[:,:,k])
                            dst.set_band_description(k+1, wls_string[k])
                else:
                    bb = rio.coords.BoundingBox(ext[0], ext[2], ext[1], ext[3]) # rasterio BoundingBox has same values as mpl imshow extent (but with order L,B,R,T)
                    transform_r, cols_r, rows_r = rio.warp.calculate_default_transform(
                        rio.crs.CRS.from_epsg(epsg),
                        rio.crs.CRS.from_epsg(epsg_tower_loc), cols, rows, *bb)
                    ometa.update({'crs': crs_tower_loc,
                                  'transform': transform_r,
                                  'width': cols_r,
                                  'height': rows_r})
                    with rio.open(crop_paths[i], 'w', **ometa) as dst:
                        for k in range(0, b):
                            rio.warp.reproject(
                                source=cube[:,:,k],
                                destination=rio.band(dst, k+1),
                                src_transform=itransforms[i],
                                src_crs=crs_utm,
                                dst_transform=transform_r,
                                dst_crs=crs_tower_loc,
                                resampling=rio.enums.Resampling.nearest)
                            dst.set_band_description(k+1, wls_string[k])
        
        flx_df = pd.DataFrame(list(zip(icos_list, crslist, lons, lats)),
                              columns=['name', 'sensorcrs', 'lon', 'lat'])
        if save_csv == True:
            flx_loc_path = self.out_dir / '{}_hsicos_crs_loc.csv'.format(self.sensor)
            flx_df.to_csv(flx_loc_path, index=False)
        self.flx_loc = gpd.GeoDataFrame(flx_df, geometry=gpd.points_from_xy(
            flx_df.lon, flx_df.lat), crs='EPSG:4326')
        
        return
        

    def _crs_calc(self, site):
        '''
        Helper function for retrieval of ICOS site geographic coordinates from
        database and calculation of matching UTM CRS for transformations.
        '''
        lon = self.icos_db.loc[self.icos_db.name == site, 'geometry'].item().x
        if lon >= -12 and lon < -6:
            crs = 32629
        elif lon >= -6 and lon < 0:
            crs = 32630
        elif lon >= 0 and lon < 6:
            crs = 32631
        elif lon >= 6 and lon < 12:
            crs = 32632
        elif lon >= 12 and lon < 18:
            crs = 32633
        elif lon >= 18 and lon < 24:
            crs = 32634
        elif lon >= 24 and lon < 30:
            crs = 32635
        else:
            raise ValueError('{}: Coordinates outside of '.format(site) +
                             'usual range for ICOS sites in Europe.')
        return crs

    def hsi_qc(self, icos_list, clip_dist = [1.0, 2.0], date = None,
               overwrite = False, save = False):
        '''
        Quality checking function for DESIS & PRISMA imagery. Checks if imagery
        covers flux site and is not located on no data edge of an image. Gen-
        erates quicklook graphics to examine potential atmospheric disturbances.
        As assessment of image quality, clouds, etc. need visual examination,
        this function does not remove entries from 'img_db' automatically. Un-
        suitable entries have to be removed manually from the table!
        
        Args:
            icos_list (string / list of strings): Abbreviation of the ICOS
                site(s) for which geometry information will be collected.
            clip_dist (list of floats, optional): 2 buffer values in kilometers.
                The second value is used for cropping the scene while the first is
                only used for plotting. PRISMA crops already use a radius of 3
                km around tower location, so values should be < 3.
            date (string, optional): A specific date (YYYY-MM-DD) can be passed to
                process a single image.
            overwrite (bool, optional): If true, existing quicklooks will be
                ignored and overwritten.
            save (bool, optional): If true, quicklooks are generated and saved
                to wdir.
        '''
        logger.info('Running hsi_qc method')

        if isinstance(icos_list, str):
            icos_list = [icos_list]
        filestatus_l = [0]*len(icos_list)

        for j,site in enumerate(icos_list):
            if date == None:
                datelist = self.img_db.loc[
                    self.img_db.name == site, 'startdate'].dt.date
            else:
                try:
                    dt.datetime.strptime(date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError('Incorrect date format, should be YYYY-MM-DD')
                datelist = self.img_db.loc[(self.img_db.name == site) & \
                                           (self.img_db.date == date),
                                           'startdate'].dt.date
            dtakes = self.img_db.loc[datelist.index, 'dataTakeID']
            filestatus = [0]*len(datelist)
            filemissing = [-99]*len(datelist)
            
            if self.sensor == 'DESIS':
                for i, dtake in enumerate(dtakes): # little loop to check imagery existence
                    filemissing[i] = len([x for x in self.img_dir.glob('*' + dtake + '*.tif')])
            elif self.sensor == 'PRISMA':
                for i, dtake in enumerate(dtakes):
                    filemissing[i] = len([x for x in self.img_dir.glob(
                        'PRS_L2D_STD_{}_{}_6km_crop.tif'.format(dtake, site))])
            
            nm_ix = np.where(np.array(filemissing) > 0)[0]
            if len(nm_ix) == 0:
                filestatus_l[j] = ['missing'] * len(datelist)
                logger.warning('No images found for site {}.'.format(site))
                continue # early end of iteration if no data found 
            
            epsg = self.flx_loc.loc[self.flx_loc.name == site, 'sensorcrs'].item()
            crs_utm = proj.CRS.from_epsg(epsg)
            flx_loc = self.icos_db.loc[self.icos_db.name == site, 'geometry'].to_crs(crs_utm)
            
            flx_roi = flx_loc.buffer(max(clip_dist)*1000)
            box_geom = geometry.box(*flx_roi.total_bounds)
            flx_roi_na = flx_loc.buffer(1000) # 1000 m buffer for checking NaN ratio
            box_geom_na = geometry.box(*flx_roi_na.total_bounds)
            
            for i, dtake in enumerate(dtakes):
                if self.sensor == 'DESIS': # after sorting: file 0 = 10-band QC image,
                # file 1 = spectral image, file 2 = RGB quicklook, file 4 = unused
                    fnames = sorted([x for x in self.img_dir.glob('*' + dtake + '*.tif')],
                                    key=lambda path: path.name[64:65]) # strange key index to get reliable sorting. don't ask...
                    qc_fn = 'DESIS-HSI-L2A-DT{}_{}_qc.png'.format(dtake.zfill(14), site)
                elif self.sensor == 'PRISMA':  # PRISMA offers no quality overviews -> 1 file per dtake
                    fnames = [x for x in self.img_dir.glob(
                        'PRS_L2D_STD_{}_{}_6km_crop.tif'.format(dtake, site))]
                    qc_fn = 'PRS_L2D_STD_{}_{}_qc.png'.format(dtake, site)
                if len(fnames) == 0: # check if file is present
                    logger.warning('The image with dataTakeID {} could not be found.'.format(dtake))
                    filestatus[i] = 'missing'
                    continue
                qc_path = self.img_dir / qc_fn
                if qc_path.is_file() & (overwrite == False):
                    logger.warning('QC for DT:{} exists already. Skipping.'.format(dtake))
                    filestatus[i] = 'good'
                    continue
                # Load & crop or create RGB quicklook
                try: # error handling if research site is not covered by input raster
                    if self.sensor == 'DESIS':
                        # fmch.desis_crop can't be used as it corrects hyperspectral digital numbers
                        # and uses a .hdr file. Both is not usable for the RGB quicklook
                        with rio.open(fnames[2], driver='GTiff') as src:
                            desis_ql, _ = riom.mask(src, shapes=[box_geom], crop=True, indexes=[3,2,1])
                        #out_ext = riop.plotting_extent(desis_ql[0], _)
                        quality_ql = np.einsum('kli->lik', desis_ql)
                        desis_cube, wls, out_ext, na_val, itrans, _ = desis_crop(
                            fnames[1], box_geom)
                        
                        na_rate_plot = (desis_cube == na_val).sum(axis=2) / desis_cube.shape[2]
                        
                        # DESIS only: Load quality information layer
                        with rio.open(fnames[0], driver='GTiff') as src:
                            desis_qi, _ = riom.mask(src, shapes=[box_geom], crop=True)
                            # 0=shadow, 1=clear land, 3=haze over land, 5=cloud over land
                        
                        # plotting params
                        pp = [2, 3, 27, 18, True, True]
                        titles = ['RGB', 'NA rate', 'Shadow', 'Haze', 'Clouds']
                    elif self.sensor == 'PRISMA':
                        with rio.open(fnames[0]) as src:
                            cube = src.read()
                            wls = [float(w) for w in list(src.descriptions)]
                            itrans = src.meta['transform'] # for plotting & cropping
                            na_val = src.nodata
                        cube = np.einsum('kli->lik', cube) 
                        out_ext = riop.plotting_extent(cube, itrans)
                        prisma_cube, otrans = _local_mask(cube, itrans, [box_geom],
                                                        crop=True, all_touched=True)
                        
                        rows, cols, b = np.shape(prisma_cube)
                        prisma_cubeT = prisma_cube.reshape(-1,b)
                        prisma_rgb = HSI2RGB(wls, prisma_cubeT, rows, cols, 50, 0.0002)
                        # transform into 8-bit RGB to match DESIS quality quicklook format
                        quality_ql = (prisma_rgb*255).round(0).astype(int)
                        
                        na_rate_plot = (prisma_cube == na_val).sum(axis=2) / prisma_cube.shape[2]
                        
                        pp = [1, 2, 18, 9, False, True]
                        titles = ['RGB', 'NA rate']
                except ValueError as e:
                    if str(e) == 'Input shapes do not overlap raster.':
                        filestatus[i] = 'geom_error'
                        continue
                    else:
                        raise
                # Check NA ratio
                na_rate_crop, _ = _local_mask(na_rate_plot[:,:,np.newaxis],
                                              itrans, [box_geom_na], crop=True)
                na_ratio = round(na_rate_crop.mean(), 2)
    
                if save == False:
                    continue
                else:
                    if na_ratio >= 0.1:
                        logger.warning('Averaged per-pixel NaN rate over all bands is {} for DataTake {} within {}km around the ICOS location!'\
                                       .format(na_ratio, dtake, min(clip_dist)))  
                        filestatus[i] = 'nodata_warn'
                    else:
                        filestatus[i] = 'good'
                        
                    fig, axes = plt.subplots(pp[0], pp[1], figsize=(pp[2], pp[3]),
                                             sharex=pp[4], sharey=pp[5])
                    divider = make_axes_locatable(axes.flat[1])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    if self.sensor == 'DESIS':                
                        axes[0,0].imshow(quality_ql, extent=out_ext)
                        axes[0,1].imshow(na_rate_plot, extent=out_ext)
                        axes[0,2].imshow(desis_qi[0,:,:], extent=out_ext)
                        axes[1,0].imshow(desis_qi[3,:,:], extent=out_ext)
                        axes[1,1].imshow(desis_qi[5,:,:], extent=out_ext)
                        for i,ax in enumerate(axes.flat[:-1]):
                            flx_loc.plot(ax=ax, marker='x', markersize=100, c='crimson')
                            flx_loc.buffer(clip_dist[0]*1000)\
                                .plot(ax=ax, fc='none', ec='crimson', linewidth=2)
                            flx_loc.buffer(clip_dist[1]*1000)\
                                .plot(ax=ax, fc='none', ec='purple', linewidth=2)
                            ax.title.set_text(titles[i])
                    elif self.sensor == 'PRISMA':                
                        axes[0].imshow(quality_ql, extent=out_ext)
                        im = axes[1].imshow(na_rate_plot, extent=out_ext)
                        fig.colorbar(im, cax=cax, orientation='vertical')
                        for i,ax in enumerate(axes.flat):
                            flx_loc.plot(ax=ax, marker='x', markersize=100, c='crimson')
                            flx_loc.buffer(clip_dist[0]*1000).plot(
                                ax=ax, fc='none', ec='crimson', lw=2)
                            ax.title.set_text(titles[i])
                    fig.savefig(qc_path, pad_inches=0, bbox_inches='tight', dpi=150)
                    plt.close(fig)
            filestatus_l[j] = filestatus
        filestatus_all = sum(filestatus_l, [])
        filestatus_df = pd.concat([self.img_db[['name', 'dataTakeID']],
                                   pd.Series(filestatus_all).rename('status')], axis=1)
        return filestatus_df


    def var_check(self, site_list):
        '''
        A function to check which ICOS station data do not include certain
        (micrometeorological) variables that are needed for flux footprint 
        modeling. If only planetary boundary layer height (PBLH) is missing,
        it can be substituted from ERA5 reanalysis data.
        
        Args:
            icos_list (string or list of strings): Abbreviation of the ICOS
                site(s) to be checked.
        '''
        # check for non-ICOS sites in site_list
        for site in site_list:
            icos_exp = self.icos_dir / 'ICOSETC_{}_FLUXES_INTERIM_L2.csv'.format(site)
            if not icos_exp.exists(): # Create file name template for single ICOS site
                icos_exp = self.icos_dir / 'ICOSETC_{}_FLUXES_L2.csv'.format(site)
                if not icos_exp.exists():
                    try:
                        icos_exp = next(self.icos_dir.glob('FLX_{}_FLUXNET*.csv'.format(site)))
                        raise RuntimeError('Site {} is NOT an ICOS site. This method should not be used for data in FLUXNET or other formats.'.format(site))
                    except StopIteration:
                        RuntimeError('No flux station data found for site {}'.format(site))
        
        cols = ['TIMESTAMP_START', 'TIMESTAMP_END', 'USTAR', 'V_SIGMA', 'MO_LENGTH', 'PBLH', 'WS', 'WD']
        l2_var_count = [0] * len(site_list)
        for i, site in enumerate(site_list):
            icos_exp = self.icos_dir / ('ICOSETC_' + site + '_FLUXES_INTERIM_L2.csv')
            if icos_exp.exists(): # Create file name template for single ICOS site
                icos_v = 'ICOSETC_' + site + '_{}_INTERIM_L2.csv'
            else:
                icos_v = 'ICOSETC_' + site + '_{}_L2.csv'
                
            flx_hh = pd.read_csv(self.icos_dir / icos_v.format('FLUXES'),
                                 date_format=self._df_icos, nrows=0,
                                 parse_dates=cols[:2],
                                 usecols=lambda x: x in cols)
            tvar = cols[2:]
            if pd.Series(tvar).isin(flx_hh.columns[2:]).all():
                l2_var_count[i] = 6
                logger.info('{}: Full L2 vars'.format(site))
            elif pd.Series(tvar[0:4]).isin(flx_hh.columns[2:]).all():
                l2_var_count[i] = 4
                logger.info('{}: Only missing WS, WD'.format(site))
            elif pd.Series(tvar[0:3]).isin(flx_hh.columns[2:]).all():
                l2_var_count[i] = 3
                logger.info('{}: Missing PBLH and/or WS, WD -> era_blh site!'.format(site))
            else:
                logger.info('{}: Missing more than 50% of required FF model variables.'\
                      .format(site))
        result = pd.DataFrame({'name': site_list,
                               'l2_var_count': l2_var_count})
        miss = result.loc[result.l2_var_count == 3, 'name'].tolist()
        prob = result.loc[result.l2_var_count < 3, 'name'].tolist()
        # Remove era_blh entries if no usable imagery is available at all
        no_usable = self.img_db.groupby('name').usable.sum()
        miss = [x for x in miss if x not in no_usable[no_usable == 0].index.tolist()]
        
        logger.warning('ICOS sites {} dont have planetary '.format(miss) +
              'boundary layer height (PBLH) data. Use the "fmch.icos_era5_get" ' +
              'method to download the data from ERA5 and use the mentioned ' +
              'stations as "era_blh" variable when using the "update_img_db" ' +
              'method.')
        if len(prob) != 0:# result any less 3: throw error:
            logger.error('ICOS sites {} have missing FF '.format(prob) +
                         'variables that will cause problems later!')
        
        return miss

#### COPERNICUS CDS/WEKEO FUNCTIONS ###########################################

    def icos_cds_get(self, icos_list, eobs = False, version = '27.0e',
                     ts = True, suffix = None):
        '''
        Requests and downloads single ERA5 PBLH pixels (~30x30 km, specific
        datetimes) or multiple E-OBS temperature & precipitation pixels (~10x10
        km, time series from 1950-01-01 to 2021-06-30) located at ICOS site(s).
        Downloaded variables are saved as .zip (E-OBS) or .nc (ERA5).
        
        Args:
            icos_list (string or list of strings): Abbreviation of the ICOS
                site(s) for which ERA 5 data will be requested.
            eobs (bool): If true, E-OBS temperature and precipitation data are
                downloaded. Otherwise, ERA5 PBLH data is downloaded.
            version (string): Version of the dataset. Check your request online
                (https://cds.climate.copernicus.eu/) to get version info.
            ts (bool, optional): If true, datetimes are provided to the ERA5
                API as timestamp values.
            suffix (str, optional): A string to be appended to the .nc file.
                Not used for E-OBS data as these cover decades.
        '''
        if suffix == None:
            suffix = self.sensor
        if isinstance(icos_list, str):
            icos_list = [icos_list]

        for i,site in enumerate(icos_list):
            c = cdsapi.Client()
            flx_loc = self.icos_db.loc[self.icos_db.name == site, 'geometry']
            if (not 30 < flx_loc.y.item() < 70) or not (-15 < flx_loc.x.item() < 45):
                raise ValueError('Coordinates for {} are out of '.format(site) +
                                 'the usual range for ICOS sites in Europe.' +
                                 ' Please investigate.')
            if eobs:
                ext = flx_loc.buffer(0.2, cap_style=3).bounds.squeeze(axis=0) # 0.2 deg!
                dataset = 'insitu-gridded-observations-europe'
                params = {
                    'format': 'zip',
                    'grid_resolution': '0.1deg',
                    'product_type': 'ensemble_mean',
                    'variable': ['maximum_temperature', 'mean_temperature',
                                 'minimum_temperature', 'precipitation_amount'],
                    'period': 'full_period',
                    'version': version,
                    'area': [ext.maxy, ext.minx, ext.miny, ext.maxx],
                    }
            else:
                epsg = self.flx_loc.loc[self.flx_loc.name == site, 'sensorcrs'].item()
                crs_utm = proj.CRS.from_epsg(epsg)
                flx_loc = flx_loc.to_crs(crs_utm)
                ext = flx_loc.buffer(100, cap_style=3).bounds.squeeze(axis=0)
                if ts == True:
                    datelist = self.img_db.loc[self.img_db.name == site,
                                               'startdate'].dt.strftime('%Y-%m-%d %H:%M').tolist()
                else:
                    datelist = [dt.datetime.date(x).isoformat() for x in self.img_db['startdate']]
                    # round to 1h to match corresp. ERA5 time slices
                    times = self.img_db.loc[self.img_db.name == site,
                                       'startdate'].dt.round('1h').dt.time.unique()
                    timelist = [x.strftime('%H:%M') for x in times]
                    timelist.sort()
                
                dataset = 'reanalysis-era5-complete'
                params = {
                    'format': 'netcdf',
                    'class': 'ea',
                    'date': datelist,
                    'expver': '1',
                    'levtype': 'sfc',
                    'param': '159.128', # PBLH
                    'stream': 'oper',
                    'type': 'an',
                    'grid': [0.25, 0.25],
                    'area': [ext.maxy, ext.minx, ext.miny, ext.maxx],
                    }
                if ts != True:
                    params['time'] = timelist # times are UTC! 
            
            request = c.retrieve(dataset, params)
            cdspath = self.icos_dir / site
            cdspath.mkdir(parents=True, exist_ok=True)
            if eobs:
                request.download(cdspath / 'e-obs_meteo_large.zip')
            else:
                request.download(cdspath / ('era5_pblh_' + suffix + '.nc'))
        return

    def icos_eobs_pet(self, icos_list, overwrite = False):
        '''
        Extracts E-OBS precipitation and temperature data for ICOS site(s).
        Resulting precip. & PET time series are saved as NetCDF.
        
        Args:
            icos_list (string or list of strings): Abbreviation of the ICOS
                site(s) for which ERA 5 data will be requested.
            overwrite (bool, optional): If true, existing NetCDFs will be
                ignored and overwritten.
        '''
        if isinstance(icos_list, str):
            icos_list = [icos_list]

        for i,site in enumerate(icos_list):
            cdspath = self.icos_dir / site
            petfile = cdspath / 'eobs_vars_pet_{}.nc'.format(site)
            if petfile.is_file() & (overwrite == False):
                logger.info('{}: PET & precipitation NetCDFs already exist. Skipping.'\
                            .format(site))
                continue
            
            variables = ['tn', 'tx', 'tg', 'rr'] # tmean, tmin, tmax, precipitation
            # caution: variable order matters: hargreaves function needs mean>min>max order
            
            datasets0 = [xr.open_dataset(list(cdspath.expanduser(
                ).glob('*'+ v +'_ens_mean_0_area_subset.nc'))[0]) for v in variables]
            ndays = len(datasets0[0].time)
            
            pet = np.zeros(ndays)
            datasets = [0]*len(datasets0)
            js = [0]*len(datasets0)
        
            eobs_tg_t = pd.to_datetime(datasets0[0].time.values).dayofyear
            if eobs_tg_t[-1] != 365:
                logger.warning('E-OBS time series for {}: '.format(site) +
                               'Last DOY is not 365 but {}'.format(eobs_tg_t[-1]))
            eobs_tg_times = pd.to_datetime(datasets0[0].time.values)
            
            flx_loc = self.icos_db.loc[self.icos_db.name == site, 'geometry']
            lat_flx = flx_loc.y.item()
            lon_flx = flx_loc.x.item()
            lat_r = pyeto.deg2rad(lat_flx)  # Convert latitude to radians
            
            # NA handling loop
            for i, (ds0, v) in enumerate(zip(datasets0, variables)):
                lats = ds0.latitude.values
                lons = ds0.longitude.values
                lat_diff = abs(lats - lat_flx) # differences list to find min. distance...
                lon_diff = abs(lons - lon_flx) # from flux tower to E-OBS grid cell
                lat = lats[np.argmin(lat_diff)]
                lon = lons[np.argmin(lon_diff)]
                
                # Slice 1 test data set for NA check. Increase indices using sorted differences
                # list to try neighboring grid cells instead of the closest cell
                ds = ds0[v].sel(longitude=lon, latitude=lat).values
                na_days = len(ds[np.isnan(ds)])
                if (na_days / ndays) > .5:
                    lat_diff_s = sorted(abs(lats - lat_flx))
                    lon_diff_s = sorted(abs(lons - lon_flx))
                    j = 0
                    while (na_days / ndays) > .5:
                        j = j + 1
                        if j == len(lats):
                            raise ValueError('No grid cell with <50% NA values' +
                                             ' found for ICOS site {}'.format(site))
                        lat = lats[lat_diff == lat_diff_s[j]]
                        ds = ds0[v].sel(longitude=lon, latitude=lat).values
                        na_days = len(ds[np.isnan(ds)])
                        if (na_days / ndays) > .5: # second check allows to change lat
                        # and lon value in turns
                            lon = lons[lon_diff == lon_diff_s[j]]
                            ds = ds0[v].sel(longitude=lon, latitude=lat).values
                            na_days = len(ds[np.isnan(ds)])
                    js[i] = j
                datasets[i] = ds
            datasets = [np.squeeze(ds).astype('float64') for ds in datasets]
                
            if (np.array(js) == 0).all():
                logger.info(f'{site}: Lat/lon coordinates suitable for all E-OBS datasets.')
            else:
                logger.info(f'{js}: Lat/lon has been shifted by {site} cells due to NA values')
    
            # using either list comprehensions ...
            sol_dec = np.array([pyeto.sol_dec(xi) for xi in eobs_tg_t])
            sha = np.array([pyeto.sunset_hour_angle(lat_r, xi) for xi in sol_dec])
            ird = np.array([pyeto.inv_rel_dist_earth_sun(xi) for xi in eobs_tg_t])
            # or lambda functions inside np.apply_along_axis to process numpy arrays row-wise
            et_rad_args = np.stack((sol_dec, sha, ird), axis=1)
            et_rad = np.apply_along_axis(lambda x: pyeto.et_rad(lat_r, x[0], x[1], x[2]), 1, et_rad_args)
            
            pet_args = np.stack(datasets[0:3] + [et_rad], axis=1)
            pet = np.apply_along_axis(lambda x: pyeto.hargreaves(x[0], x[1], x[2], x[3]), 1, pet_args)
            
            # Build xarrays for .nc saving
            pet_ds = xr.Dataset(
                data_vars={'pet': (('time'), pet,
                                   {'unit': 'mm', 'long_name': 'potential evapotranspiration'}),
                           'pre': (('time'), datasets[3],
                                   {'unit': 'mm', 'long_name': 'precipitation'}),
                           'tn': (('time'), datasets[0],
                                  {'unit': 'degC', 'long_name': 'minimum temperature'}),
                           'tx': (('time'), datasets[1],
                                  {'unit': 'degC', 'long_name': 'maximum temperature'}),
                           'tg': (('time'), datasets[2],
                                  {'unit': 'degC', 'long_name': 'mean temperature'})},
                coords={'time': eobs_tg_times},
                attrs={'creation_date': dt.datetime.now().isoformat(),
                       'source': 'E-OBS v25.0e',
                       'author': 'Floris Hermanns',
                       'email': 'floris.hermanns@posteo.de',
                       'comment': 'created with the Hargreaves model implemented in the PyETo module'})
            pet_ds.to_netcdf(petfile)
        return

    
    def _icos_era5_pblh(self, icos_site, nc_name = None):
        '''
        Extracts ERA5 data matching the acquisition timestamps of available
        hyperspectral imagery for a single ICOS site.
        
        Args:
            icos_site (string): Abbreviation of the ICOS site for which ERA 5
                data will be extracted.
            nc_name (string, optional): Name of the NetCDF file that stores
                ERA5 PBLH time series. The file must only contain one pixel.
        '''
        if nc_name == None:
            nc_name = 'era5_pblh_{}.nc'.format(self.sensor)
        era5 = xr.open_dataset(self.icos_dir / icos_site / nc_name)
        
        era5_times = pd.to_datetime(era5.blh.time.values)
        img_db_site = self.img_db[self.img_db.name == icos_site]
        pblh = [0]*len(img_db_site)
        for i,idate in enumerate(img_db_site.startdate.dt.round('1h')):
            ts = nearest(era5_times, idate)
            # in case of nan values, cdsapi will download additional expver version
            if 'expver' in era5.dims:
                v = era5.coords['expver'].values # use last version (!= 1)
                val = era5.blh.sel(time=ts, expver=v[0]).item()
                if np.isnan(val):
                    val = era5.blh.sel(time=ts, expver=v[1]).item()
            else:
                val = era5.blh.sel(time=ts).item()
            pblh[i] = val
        if np.isnan(pblh).any():
            raise ValueError('NaN value detected in ERA5 derived PBLH values.')
        return pblh

    def update_img_db(self, img_csv, era_blh = None, nc_name = None, save = False): #only docstring yet corrected for spyder help window formatting
        '''
        Update 'img_db' data frame after manual modifications have been made
        following visual examination of QC images:
        
        - Remove unsuitable imagery.
        - Add BLH values to img_db if missing.
        
        Args:
            img_csv (string):
                Name of the metadata CSV that contains key info about the
                imagery to be processed. Required columns are:
                
                - 'name': Name or abbr. of the corresp. ICOS site
                - 'startdate': Timestamp of image acquisition (YYYY-MM-DD hh:mm:ss)
                - 'dataTakeID': datatake ID of the scene
                - 'clouds': commentary field to mark missing data, etc.
                - 'era_pblh': planetary boundary layer height value from ERA5
                    reanalysis. Only used when PBLH is missing from ICOS L2 data.
                - 'usable': integer values for results of imagery inspection.
                    0 = unusable, 1 = usable, 2 = maybe usable (requires FF
                    inspection), 3 = missing flux data
            era_blh (list of strings, optional):
                ICOS abbreviations of sites which do not provide boundary layer
                height (BLH) values for FF estimation. BLH values will be
                imported from ERA5 if available. If None, this method will only
                remove unusable imagery.
            nc_name (string, optional):
                Name of the NetCDF file that stores ERA5 PBLH time series. The
                file must only contain one pixel. If None, default ICOS file
                name structure is used.
            save (bool, optional):
                If true, the cleaned imagery data frame is saved as CSV.
        '''
        old_img_db = pd.read_csv(
            self.icos_dir.parent / img_csv, parse_dates=['startdate'],
            date_format=self._df, dtype={'dataTakeID': str}).reset_index(drop=True)
        
        if all(x in old_img_db.columns for x in \
               ['name', 'startdate', 'dataTakeID', 'clouds', 'usable']):
            pass
        else:
            raise ValueError('Missing columns in "img_db". Please prepare the '\
                             'data frame according to documentation.')
        ppi_df = self.img_db[['name', 'dataTakeID', 'ppi_file']]
        checkdate = old_img_db[['name', 'startdate']].select_dtypes(include=[np.datetime64])
        if checkdate.shape[1] == 0:
            raise ValueError('Incorrect datetime format, should be np.datetime64.')
        
        old_img_db['date'] = [dt.datetime.date(x).isoformat() for x in old_img_db['startdate']]
        old_img_db['icostime'] = (old_img_db.startdate +
                                   pd.Timedelta(hours=1)).dt.round('30min').dt.time
        # only keep usable & maybe usable imagery entries
        self.img_db = old_img_db[old_img_db.usable.between(1,2)].reset_index(drop=True)
        
        if era_blh != None:
            self._era_blh = era_blh
            if (isinstance(era_blh, list) and all(isinstance(x, str) for x in era_blh)):
                pass
            else:
                raise ValueError('Incorrect era_blh specification (list of strings required).')
            
            # add PBLH values from ERA5 data
            self.img_db['era_pblh'] = 0
            for site in self._era_blh:
                pblh_values = self._icos_era5_pblh(site, nc_name = nc_name)
                self.img_db.loc[self.img_db.name == site, 'era_pblh'] = pblh_values
        else:
            self._era_blh = []
            
        self.img_db = pd.merge(self.img_db, ppi_df, how='left',
                               on=['name', 'dataTakeID'], suffixes=('', '_y'))
        self.img_db.drop(self.img_db.filter(regex='_y$').columns, axis=1, inplace=True)
        
        if save == True:
            self.img_db.to_csv(self.icos_dir.parent / (img_csv[:-4] + '_f.csv'), index=False)
        icos_list = self.img_db.name.unique().tolist()
        
        return icos_list

### ICOS FUNCTIONS ############################################################

    def _ppfd_check(self, site_list):
        '''
        A helper function to find out which ICOS station data does not include
        photosynthetic photon flux density (PPFD_IN), or neither PPFD or incom-
        ing shortwave radiation (SW_IN_F). If PPFD is not available, it is es-
        timated from SW. If both are missing, the site is removed from further
        calculations.
        '''
        no_ppfd = []
        no_rad = []
        cols = ['TIMESTAMP_START', 'TIMESTAMP_END', 'PPFD_IN', 'SW_IN_F']
        for i, site in enumerate(site_list):
            flux_exp = self.icos_dir / 'ICOSETC_{}_FLUXNET_HH_INTERIM_L2.csv'.format(site)
            if flux_exp.exists(): # Create file name template for single ICOS site
                pass
            else:
                flux_exp = self.icos_dir / 'ICOSETC_{}_FLUXNET_HH_L2.csv'.format(site)
                if flux_exp.exists():
                    pass
                else:
                    try:
                        flux_exp = next(self.icos_dir.glob('FLX_{}_FLUXNET*.csv'.format(site)))
                    except StopIteration:
                        raise RuntimeError('{}: No FLUXNET csv found.'.format(site))
                
            flx_gpp = pd.read_csv(flux_exp, date_format=self._df_icos,
                                  nrows=0, parse_dates=cols[:2],
                                  usecols=lambda x: x in cols)
            tvar = ['PPFD_IN', 'SW_IN_F']
            if ~pd.Series(tvar[0]).isin(flx_gpp.columns).all(): # if PPFD not in columns
                no_ppfd.append(site)
            elif ~pd.Series(tvar).isin(flx_gpp.columns).all(): # if neither variable is in columns
                no_rad.append(site)
        if len(no_ppfd) > 0:
            logger.info('ICOS site(s) {} dont have PPFD '.format(no_ppfd) +
                        'measurements in their FLUXNET dataset. PAR will ' +
                        'be calculated from SW_IN.')
        if len(no_rad) > 0:
            logger.warning('ICOS site(s) {} have neither '.format(no_ppfd) +
                           'PPFD nor SW_IN data and will be removed.')
        return no_ppfd, no_rad


    def _icos_height_aux(self, flx_anc, icos_site, D_avg = False):
        '''
        Helper function that cycles through the different variants of obtaining
        vegetation height information from ICOS L2 data: By default, all obser-
        vations of the 75th percentile will be used. If not available, mean
        values or single observations will be used or averaged if observations
        are only available per species (HEIGHTC_SPP) and not for the whole
        canopy.
        '''
        flx_anc_w = flx_anc[flx_anc.VARIABLE_GROUP == 'GRP_HEIGHTC'] \
            .pivot(index='GROUP_ID', columns='VARIABLE', values='DATAVALUE')
        if flx_anc_w.HEIGHTC_STATISTIC.isin(['75th Percentile']).any() \
            & ~flx_anc.VARIABLE.isin(['HEIGHTC_SPP']).any():
            D_gid0 = flx_anc_w.loc[
                (flx_anc_w.HEIGHTC_STATISTIC == '75th Percentile'), :].index
        elif flx_anc_w.HEIGHTC_STATISTIC.isin(['75th Percentile']).any() \
            & flx_anc.VARIABLE.isin(['HEIGHTC_SPP']).any():
            if flx_anc_w.HEIGHTC_SPP.isnull().any():
                D_gid0 = flx_anc_w.loc[
                    (flx_anc_w.HEIGHTC_STATISTIC == '75th Percentile') \
                    & flx_anc_w.HEIGHTC_SPP.isnull(), :].index
        # DE-Hai only has SPP entries from 1 date -> shortcut to D
        elif flx_anc_w.HEIGHTC_STATISTIC.isin(['Mean', 'Single observation']).any() \
            & ~flx_anc.VARIABLE.isin(['HEIGHTC_SPP']).any():
            D_gid0 = flx_anc_w.loc[flx_anc_w.HEIGHTC_STATISTIC.isin(
                ['Mean', 'Single observation']), :].index
            logger.info('Using mean instead of P75 values for site {}'.format(icos_site))
        elif flx_anc_w.HEIGHTC_STATISTIC.isin(['Mean', 'Single observation']).any() \
            & flx_anc.VARIABLE.isin(['HEIGHTC_SPP']).any():
            if flx_anc_w.HEIGHTC_SPP.isnull().any():
                D_gid0 = flx_anc_w.loc[
                    flx_anc_w.HEIGHTC_STATISTIC.isin(['Mean', 'Single observation']) \
                    & flx_anc_w.HEIGHTC_SPP.isnull() , :].index
                logger.info('Using mean instead of P75 values for site {}'.format(icos_site))
            else:
                D_gid0 = flx_anc_w.loc[(flx_anc_w.HEIGHTC_STATISTIC == 'Mean') \
                                       & flx_anc_w.HEIGHTC_SPP.notnull() , :].index
                D_avg = True
                logger.info('Using SPP means of canopy height for site {}'\
                            .format(icos_site))
        else:
            D_gid0 = []
        if len(D_gid0) == 0:
            raise KeyError('No usable mean or P75 values for D found in ICOS data.')
        else:
            D_hdates = pd.to_datetime(
                flx_anc_w.loc[flx_anc_w.index.isin(D_gid0), 'HEIGHTC_DATE']).dt.date
            
        return flx_anc_w, D_hdates, D_avg
    

    def _load_icos_subset(self, icos_site, datelist, fluxvars, aggregate = 'na',
                          ppfd_missing = [], zonal = False):
        '''
        Reads a number of ICOS level 2 CSV files (FLUXES, FLUXNET_HH,
        ANCILLARY, INST, if applicable METEO) to extract information on meteo-
        rological variables required for flux footprint modeling (WS, WD, PBLH,
        MO_LENGTH, V_SIGMA, USTAR) and ecosystem productivity estimates. Data
        are reduced to the time of interest, i.e. the days for which hyperspec-
        tral imagery are available.
        If zonal statistics were selected, only productivity estimates will be
        exported from L2 data.
        
        Args:
            icos_site (string): Abbreviation of the ICOS site.
            datelist (pandas.Series): Dates of available imagery for the
                selected ICOS site.
            fluxvars (list of strings): ICOS variable names of ecosystem fluxes
                (CO2, radiation, ...) to be imported.
            aggregate (string): Type of aggregation of daily ICOS values, either
                "na", "mean" or "sum".
            ppfd_missing (list of strings, optional): ICOS sites for which PPFD
                variable is missing and will be derived from SW_IN instead.
            zonal (bool, optional): If true, no footprints will be calculated
                and the import of ICOS data will be restricted to fluxes only
                (no meteorological variables required).
        '''
        timelist = self.img_db.loc[datelist.index, 'icostime']
        datetimes = pd.to_datetime(datelist.astype(str) + " " + timelist.astype(str))
        D_avg = False
        D_default = False
        ZM = [-9999]*len(datelist)
        missd = [pd.Timestamp('19000101 00:00:00')] # return dummy
        
        dc = ['TIMESTAMP_START', 'TIMESTAMP_END']
        cols_gpp = dc + fluxvars
        flux_exp = self.icos_dir / ('ICOSETC_' + icos_site + '_FLUXNET_HH_INTERIM_L2.csv')
        if flux_exp.exists(): # Create file name template for single ICOS site
            icos_v = 'ICOSETC_' + icos_site + '_{}_INTERIM_L2.csv'
        else:
            flux_exp = self.icos_dir / 'ICOSETC_{}_FLUXNET_HH_L2.csv'.format(icos_site)
            if flux_exp.exists():
                icos_v = 'ICOSETC_' + icos_site + '_{}_L2.csv'
            else:
                flux_exp = next(self.icos_dir.glob('FLX_{}_FLUXNET*.csv'.format(icos_site)))
                if zonal == False:
                    raise RuntimeError('''{}: Site data only available in FLUXNET
                                       format but zonal is False. Calculation of
                                       FFs is only possible for data in ICOS format.''')
            
        # Import GPP percentiles from ensemble (daytime & nighttime)
        flx_gpp = pd.read_csv(flux_exp, date_format=self._df_icos,
                              parse_dates=dc, usecols=lambda x: x in cols_gpp)
    
        # Reference date to deal with missing flux data
        # not so elegant... maybe sort unsuitable imagery out before?
        #dl_dummy = datelist.reset_index(drop=True)
        #beforeicosstart = dl_dummy[dl_dummy < flx_gpp['TIMESTAMP_START'].dt.date.iloc[0]].index
        #aftericosend = dl_dummy[flx_gpp['TIMESTAMP_START'].dt.date.iloc[-1] < dl_dummy].index
        #self._no_flx_data_ix = beforeicosstart.union(aftericosend)
        
        # SW_IN is always added to the output data frame. PPFD_IN is converted to
        # PAR if available.
        if icos_site not in ppfd_missing:
            flx_gpp_temp0 = flx_gpp.loc[flx_gpp.TIMESTAMP_END.dt.date.isin(datelist),
                                       cols_gpp[1:]]
            flx_gpp_temp0['PAR'] = flx_gpp_temp0.PPFD_IN / 4.57
            flx_gpp_temp0.drop('PPFD_IN', axis=1, inplace=True)
            fluxvars = [x for x in fluxvars if x != 'PPFD_IN'] + ['PAR']
        else: # If PPFD not available, it's estimated from SW_IN instead.
            fluxvars = [x for x in fluxvars if x != 'PPFD_IN']
            flx_gpp_temp0 = flx_gpp.loc[flx_gpp.TIMESTAMP_END.dt.date.isin(datelist),
                                       ['TIMESTAMP_END'] + fluxvars]
            flx_gpp_temp0['PAR'] = flx_gpp_temp0.SW_IN_F * 0.47 # average PAR/SI ratio
            fluxvars = fluxvars + ['PAR']
        '''ICOS PPFD 400-700 nm: http://archive.sciendo.com/INTAG/intag.2017.32.issue-4/intag-2017-0049/intag-2017-0049.pdf
        1 W/m2 ~ 4.57 micromole/m2/s (in Foken 2008 "Micrometeorology" recherchieren)
        flx_geom_gdf contains the variables 'PPFD_IN', 'SW_IN_F' for UPW calculations
        SW_IN can be used to calculate PAR when missing: https://doi.org/10.2134/agronj1984.00021962007600060018x
        PAR/SW_IN = 0.47 -> mean from mid-latitude studies referenced in Walker (2005)'''
        # In any case, output DF will contain 'SW_IN_F' & 'PAR' but not 'PPFD_IN'
        flx_gpp_temp0.loc[flx_gpp_temp0.PAR < -100, 'PAR'] = -9999
        
        # If PAR colum is present but value is NA (and SW_IN is present), PAR will also be calculated from SW_IN
        if (flx_gpp_temp0.PAR == -9999).any():
            correct_rows = flx_gpp_temp0.loc[
                (flx_gpp_temp0.PAR == -9999) & (flx_gpp_temp0.SW_IN_F > 0)].index
            flx_gpp_temp0.loc[correct_rows, 'PAR'] = flx_gpp_temp0.loc[correct_rows, 'SW_IN_F'] * 0.47
            
        # to exclude invalid values from potential aggregation
        exclude_rows = flx_gpp_temp0[(flx_gpp_temp0.loc[:,fluxvars[:9]] == -9999).any(axis=1)].index
        # 'time's in timelist are already tz:CET! -> reduce to 1 row per HSI
        flx_gpp_temp = flx_gpp_temp0.loc[flx_gpp_temp0.TIMESTAMP_END.isin(datetimes), :]
        
        # only certain flux columns (fluxvars[:9]) are aggregated, QC columns and radiation fluxes are excluded (unaggregated PAR required for UPW transformation)
        if aggregate == 'mean': # for mean, night values have to be dropped for GPP
            flx_gpp_aggr = flx_gpp_temp0.drop(exclude_rows, axis=0)\
                .loc[(flx_gpp_temp0.PAR > 0) | (flx_gpp_temp0.PAR == -9999), # keep PAR NA values while filtering out night time values
                     fluxvars[:9]].groupby(
                         flx_gpp_temp0.TIMESTAMP_END.dt.date, sort=False).agg(
                             {var: 'mean' for var in fluxvars[:9]})
            fluxvars[:9] = ['{}_mean'.format(x) for x in fluxvars[:9]]
        elif aggregate == 'sum':
            flx_gpp_aggr = flx_gpp_temp0.drop(exclude_rows, axis=0)\
                .loc[:, fluxvars[:9]].groupby(
                    flx_gpp_temp0.TIMESTAMP_END.dt.date, sort=False).agg(
                        {var: 'sum' for var in fluxvars[:9]})
            fluxvars[:9] = ['{}_sum'.format(x) for x in fluxvars[:9]]

        if aggregate != 'na':
            flx_gpp_aggr.columns = fluxvars[:9]
            # join aggregated fluxes with unaggregated columns and add timestamp
            flx_gpp_aggr = pd.merge(flx_gpp_aggr, flx_gpp_temp[fluxvars[9:]], how='left',
                                    left_index=True, right_on=flx_gpp_temp.TIMESTAMP_END.dt.date)
            flx_gpp_aggr.insert(0, 'TIMESTAMP_END', flx_gpp_temp.TIMESTAMP_END.values)
            flx_gpp_temp = flx_gpp_aggr.drop('key_0', axis=1)
        # add missing dates back in
        if len(datelist) > flx_gpp_temp.shape[0]:
            ds1 = set(datetimes.tolist())
            ds2 = set(flx_gpp_temp.TIMESTAMP_END)
            missd = list(ds1.difference(ds2))
            add = pd.DataFrame(np.full((len(missd), len(fluxvars)+1), -9999))
            add.columns = [dc[1]] + fluxvars
            add.iloc[:, 0] = missd
            flx_gpp_temp = pd.concat([flx_gpp_temp, add], axis=0)
            flx_gpp_temp.sort_values(by=dc[1], inplace=True)
            logger.debug('date(s) missing in flx_gpp_temp: {}'.format(missd))
            logger.debug('added NA row(s): {}'.format(add))

        if zonal: # Only insert TS and return before FF calc.
            icos_subset = flx_gpp_temp.reset_index(drop=True)
            return icos_subset, ZM, fluxvars, missd
        
        cols_flx = ['USTAR', 'V_SIGMA', 'MO_LENGTH', 'PBLH', 'WS', 'WD']
        cols = dc + cols_flx
        igbp_class = self.icos_db.loc[self.icos_db.name == icos_site, 'ecosystem'].item()
        # Import micrometeo. data and NEE
        flx = pd.read_csv(self.icos_dir / icos_v.format('FLUXES'),
                          date_format=self._df_icos,
                          parse_dates=dc, usecols=lambda x: x in cols)
        
        # Import 75th percentile of canopy height
        flx_anc = pd.read_csv(self.icos_dir / icos_v.format('ANCILLARY'),
                              encoding = 'unicode_escape')
        if igbp_class in ['DBF', 'ENF', 'EBF', 'MF']:
            if flx_anc.VARIABLE_GROUP.isin(['GRP_TREE_HEIGHT']).any():
                flx_anc_w = flx_anc[flx_anc.VARIABLE_GROUP == 'GRP_TREE_HEIGHT'] \
                    .pivot(index='GROUP_ID', columns='VARIABLE', values='DATAVALUE')
                # Serach for 75th perc. where HEIGHTC_SPP is NaN (to get value for whole stand)
                D_gid0 = flx_anc_w.loc[(flx_anc_w.HEIGHTC_STATISTIC == '75th Percentile')
                                       & flx_anc_w.HEIGHTC_SPP.isnull() , :].index
                try:
                    D_hdates = pd.to_datetime(flx_anc_w.loc[
                        flx_anc_w.index.isin(D_gid0), 'HEIGHTC_DATE_START']).dt.date
                    if D_hdates.isnull().any():
                        logger.error('{}: Inconsistent variable names detected in ICOS metadata. Workaround might cause problems.'.format(icos_site))
                        nix = D_hdates[D_hdates.isnull()].index.values
                        D_hdates[nix] = pd.to_datetime(flx_anc_w.loc[
                            flx_anc_w.index.isin(nix), 'HEIGHTC_DATE']).dt.date
                except KeyError:
                    D_hdates = pd.to_datetime(flx_anc_w.loc[
                        flx_anc_w.index.isin(D_gid0), 'HEIGHTC_DATE']).dt.date

            elif flx_anc.VARIABLE_GROUP.isin(['GRP_HEIGHTC']).any(): # Sonderfall DE-Hai
                flx_anc_w, D_hdates, D_avg = self._icos_height_aux(flx_anc, icos_site)
            else:
                raise KeyError('Ancillary file for ICOS site {} '.format(icos_site) +
                               'holds no data about vegetation height.')
        
        elif igbp_class in ['CRO', 'GRA', 'OSH', 'CSH', 'SAV', 'WSA']:
            flx_anc_w, D_hdates, D_avg = self._icos_height_aux(flx_anc, icos_site)
            
        elif igbp_class == 'WET':
            D_default = True
            logger.info('Using default value for canopy height for wetland' +
                        ' site {} (D = 0.1m)'.format(icos_site))
            D = 0.1 # default low vegetation height for bogs as no height is reported
        
        #elif igbp_class in ['OSH', 'CSH']:
        #    D_default = True
        #    logger.info('Using default value for canopy height for' +
        #                ' shrubland site {} (D = 1.5m)'.format(icos_site))
        #    D = 1.5
        # Might add more classes, e.g. ES-Cnd is a WSA (woody savanna) site but ICOS L2 data are not available yet
        if D_avg == True:
            D = flx_anc_w.loc[D_hdates.index, 'HEIGHTC'].astype(float).mean()
            
        # Import sensor height
        flx_ins = pd.read_csv(self.icos_dir / icos_v.format('INST'), on_bad_lines='skip')
            
        flx_ins_w = flx_ins[flx_ins.VARIABLE_GROUP == 'GRP_INSTOM'] \
            .pivot(index='GROUP_ID', columns='VARIABLE', values='DATAVALUE')
    
        # Filter for EC sensor installation IDs and extract dates
        if len(flx_ins_w.loc[flx_ins_w.INSTOM_MODEL.str.contains('LI-COR'), 'INSTOM_HEIGHT']) > 0:
            senstr = 'LI-COR'
        elif len(flx_ins_w.loc[flx_ins_w.INSTOM_MODEL.str.contains('Campbell EC'), 'INSTOM_HEIGHT']) > 0:
            senstr = 'Campbell EC'
        else:
            raise KeyError('Unknown EC sensor model. Please check your ICOS ' +
                           'instrument metadata.')
        Z_gid0 = flx_ins_w.loc[flx_ins_w.INSTOM_MODEL.str.contains(senstr) & \
                               (flx_ins_w.INSTOM_TYPE == 'Installation') , :].index        
        Z_hdates0 = pd.to_datetime(flx_ins_w.loc[flx_ins_w.index.isin(Z_gid0), 'INSTOM_DATE']).dt.date
        # Swap index and gid values and truncate installation dates after image acq. date.
        Z_hdates = pd.Series(Z_hdates0.index.values, index=Z_hdates0).sort_index() # truncate requires sorted index
        # now I want to compare the entries to get height meas. from a date closest
        # to image acq. date (for canopy height) or the last entry before acq.
        # date (for sensor height).    
        for i, idate in enumerate(datelist):
            if idate in [x.date() for x in missd]:
                continue
            else:
                if (D_avg == True) | (D_default == True):
                    pass
                else:
                    D_gid = D_hdates[D_hdates == nearest(D_hdates.values, idate)].index[-1] # Find nearest date and in case of multiple entries, choose last
                    D = flx_anc_w.loc[flx_anc_w.index == D_gid, 'HEIGHTC'].astype(float).item()
                try:
                    Z_gid = Z_hdates.truncate(after=idate)[-1] # Use last valid installation date.
                except IndexError: # if image date is before official installation date, use first installation date
                    Z_gid = Z_hdates[0]
                Z = flx_ins_w.loc[flx_ins_w.index == Z_gid, 'INSTOM_HEIGHT'].astype(float).item()
                if D > Z:
                    raise ValueError('Canopy height D > sensor height Z. Please check your ICOS ancillary metadata.')
                ZM[i] = Z-D
        
        # Reduce DF to dates of interest (all records for days of interest) and concatenate
        if icos_site in self._era_blh:
            flx0 = flx.loc[flx.TIMESTAMP_END.dt.date.isin(datelist),
                           ['TIMESTAMP_END'] + cols_flx[:3]]
            flx_met0 = pd.read_csv(self.icos_dir / icos_v.format('METEO'),
                                   date_format=self._df_icos, parse_dates=dc)
            flx_met = flx_met0.loc[flx_met0.TIMESTAMP_END.dt.date.isin(datelist),
                                   cols_flx[4:]]
            flx_temp = pd.concat([flx0.reset_index(drop=True),
                                  flx_met.reset_index(drop=True)], axis=1)
        else:
            flx_temp = flx.loc[flx.TIMESTAMP_END.dt.date.isin(datelist),
                               cols[1:]]
        # flx_temp is not aggregated as micromet. data is only required for FF modeling
        icos_subset = pd.merge(flx_gpp_temp.reset_index(drop=True),
                               flx_temp.reset_index(drop=True),
                               how='left', on='TIMESTAMP_END')
        icos_subset.replace(np.nan, -9999, inplace=True)
        # Final ICOS subset only contains 1 row per HSI
        return icos_subset, ZM, fluxvars, missd


### FF & CROPPING FUNCTIONS ###################################################
    
    def _model_geoms(self, icos_site, datelist, icos_subset, response, ZM,
                     fluxvars, missd, zonal = False, upw = False):
        '''
        Calculates geometries matching the timestamps of hyperspectral imagery
        in 'img_db' (either flux footprints or buffers). Uses the external
        'FFP' function from the flux footprint modeling module from Kljun et al.
        
        Args:
            icos_site (string): Abbreviation of the ICOS site.
            datelist (pandas.Series): Dates of available imagery for the
                selected ICOS site.
            icos_subset (pandas.DataFrame): Subset of ICOS L2 data cut to the
                dates of interest (single site!). Contains fluxes and if
                necessary micrometeorological variables for FF modeling.
            ZM (list): Differences between observation & vegetation height.
            fluxvars (list of strings): ICOS variable names of ecosystem fluxes
                (CO2, radiation, ...) to be imported.
            missd (list of pd.Timestamps): dates from datelist for which ICOS
                data is completely missing.
            zonal (bool, optional): If true, zonal statistics (buffer value
                depending on ecosystem) will be calculated instead of FFs.
        '''
        timelist = self.img_db.loc[datelist.index, 'icostime']
        dtakes = self.img_db.loc[datelist.index, 'dataTakeID']
        ff_params = [0]*len(datelist)
        
        # Project to Lambert Azimuthal Equal Area (LAEA) for collection of european geometries in a single gdf
        crs_lam = proj.CRS.from_epsg('3035')
        flx_loc = self.icos_db.loc[self.icos_db.name == icos_site,
                                   'geometry'].to_crs(crs_lam)
        
        if zonal:
            ecosystem = self.icos_db.loc[self.icos_db.name == icos_site, 'ecosystem'].item()
            if ecosystem in ['MF', 'DBF', 'EBF', 'ENF']:
                zr = 80
            elif ecosystem in ['SAV', 'WSA']:
                zr = 60
            elif ecosystem in ['GRA', 'OSH', 'CSH']:
                zr = 50
            elif ecosystem == 'WET':
                zr = 40
            elif ecosystem == 'CRO':
                zr = 30
            else:
                raise ValueError('Ecosystem {} not supported '.format(ecosystem) +
                                 'Please update codebase.')
            geoms = [flx_loc.buffer(zr).item()] * len(timelist)
            checkvar = [response]
        else:
            geoms = [None]*len(datelist)
            checkvar = ['WS', 'PBLH', 'MO_LENGTH', 'V_SIGMA', 'USTAR', 'WD', response]
        
        #transformer = proj.Transformer.from_crs(4326, crs_utm, always_xy=True)
        #x, y = transformer.transform(lon, lat)
        for i,date in enumerate(datelist):
            ff_params[i] = icos_subset.loc[
                icos_subset.TIMESTAMP_END.dt.date == date, :].squeeze(axis=0) # squeeze creates series instead of 1D DF
            if icos_site in self._era_blh:
                pblh = pd.Series([self.img_db.loc[datelist.index[i],
                                              'era_pblh']], index=['PBLH'])
                ff_params[i] = ff_params[i].append(pblh)
            # In case of Nodata values in ICOS data, no FF can be calculated
            check9999 = ff_params[i][checkvar].isin([-9999]).any()
            checkdate = (date in [x.date() for x in missd])
            if check9999 | checkdate:
                self.img_db.loc[(self.img_db.name == icos_site) &
                                (self.img_db.dataTakeID == dtakes.iloc[i]),
                                'clouds'] = 'icos_na'
                logger.debug('{}: No ICOS data available for {} image {} from {}.'\
                             .format(icos_site, self.sensor, dtakes.iloc[i],
                                       datelist.iloc[i]))
                #ff_params[i] = pd.Series(data=[0]*len(fluxvars), index=fluxvars) # probably unneeded
            # Estimate 80th (and 50th) percentile of flux footprint with Kljun model
            if zonal:
                continue
            else:
                ff = ffp.FFP(zm=ZM[i], umean=ff_params[i].WS, h=ff_params[i].PBLH,
                                 ol=ff_params[i].MO_LENGTH, sigmav=ff_params[i].V_SIGMA,
                                 ustar=ff_params[i].USTAR, wind_dir=ff_params[i].WD,
                                 rs=[50., 80.], fig=0)
            
                # Compute FF coordinates & create geometry. Uses 80% FF contribution area (index 1)
                xs = [round(xoff + flx_loc.x.item(), 2) for xoff in ff['xr'][1]]
                ys = [round(yoff + flx_loc.y.item(), 2) for yoff in ff['yr'][1]]
                geoms[i] = Polygon(zip(xs, ys))
        
        # Prepare DF for merging with geoinformation.
        img_df = self.img_db.loc[datelist.index,
                                 ['name', 'startdate', 'date', 'dataTakeID',
                                  'clouds', 'ppi_file']].reset_index(drop=True)
        # Create single geopandas object holding information about geometry and observations
        geom_gdf = gpd.GeoDataFrame(data=img_df, geometry=gpd.GeoSeries(
            geoms, crs=crs_lam))
        # Add data about GPP, NEE
        flx_list = [0]*len(timelist)
        for i,pset in enumerate(ff_params):
            flx_list[i] = pset.loc[fluxvars]

        flx_df = pd.concat(flx_list, axis=1).T.reset_index(drop=True)
        flx_geom_gdf = pd.concat([geom_gdf, flx_df], axis=1)
        
        ix = flx_geom_gdf.columns.get_loc(fluxvars[0]) # get index of first prod variable
        flx_geom_gdf.iloc[:, ix:ix+len(fluxvars)] = flx_geom_gdf.iloc[
            :, ix:ix+len(fluxvars)].apply(pd.to_numeric, errors='ignore')
        
        # Notify in case of lower quality values in important variables
        ql_flag = 0
        for var in ['NEE_VUT_50_QC', 'SW_IN_F_QC']:
            quality = 'medium'
            for flag in [2, 3]:
                if (flx_geom_gdf[var] == flag).any():
                    if flag == 3:
                        quality = 'poor'
                        ql_flag = 1
                    ix = flx_geom_gdf[flx_geom_gdf[var] == flag].index
                    if len(ix) == 1:
                        ix = ix.item()
                    else:
                        ix = ix.tolist()
                    logger.debug('{}: {} data quality for {} dataTakeID(s) {} '\
                                 .format(icos_site, var, self.sensor,
                                         dtakes.iloc[ix]) + 'is {} ({})'\
                                     .format(quality, flag))
        
        # Sort out imagery outside the ICOS record period and where flux or hyperspectral data have NAs
        # has to be done here so that the gdf is compatible with mask_params in _crop_data_2_geoms
        nna = len(flx_geom_gdf[flx_geom_gdf.clouds == 'icos_na'])
        if nna == len(flx_geom_gdf):
            logger.warning(f'{icos_site}: All imagery acquisition dates have no'
                           ' ICOS data available. Station will be missing in'
                           ' output DF.')
        elif nna > 0:
            logger.info(f'{icos_site}: {nna} images with missing ICOS data were removed. ')
           
        flx_geom_gdf = flx_geom_gdf.loc[flx_geom_gdf[response] != 0, :] # remove zero GPP entries
        flx_geom_gdf = flx_geom_gdf[flx_geom_gdf.clouds != 'icos_na'].reset_index(drop=True)
        if upw == True:
            flx_geom_gdf = flx_geom_gdf.loc[flx_geom_gdf.PAR != -9999, :].reset_index(drop=True)
        
        return flx_geom_gdf, ql_flag, nna
    
    def _crop_data_2_geoms(self, icos_site, flx_geom_gdf, mask_params, dimred = None,
                           sr = 'vnir', upw = False, save_plot = False):
        '''
        Crops HSI / dimension-reduced HSI to geometries of interest and averages
        resulting pixels per band. Cropped imagery can be saved as GeoTIFF.
        For non-DR imagery, the NIRvP (structural GPP proxy [1]) is calculated.
        Additionally, RGB representations of the tower surroundings can be plotted.
        
        [1] Dechant, Benjamin, et al. (2022): NIRVP: A robust structural proxy
        for sun-induced chlorophyll fluorescence and photosynthesis across
        scales. Remote Sens. Environ., 268, 112763, DOI:10.1016/j.rse.2021.112763
        
        Args:
            icos_site (string): Abbreviation of the ICOS site.
            flx_geom_gdf (geopandas.GeoDataFrame): Flux measurements that match
                the observation times of the hyperspectral imagery including
                geometries for cropping (zonal or FF). Subset for 1 site!
            mask_params (pandas.DataFrame): Parameters for masking non-
                vegetation pixels. Must be determined beforehand by visual
                analysis. Length must equal that of flx_geom_gdf. Subset for 1 site!
            dimred (string, optional): If DR has been applied, dimred describes
                the DR method and number of components, e.g. 'AE04', 'SiVM10',
                etc. Used for constructing (DR) imagery folder name.
            sr (string, optional): Abbreviation of the spectral range within
                which HSI will be used, either 'vis', 'vnir' or 'vswir' (the
                latter is only available for PRISMA data).
            upw (bool, optional): If true, reflectance values will be cropped
                to visible light range and multiplied with PAR from ICOS sensor
                data resulting in 400-700 nm upwelling radiation (UPW).
            save_plot (bool, optional): If true, ICOS site surroundings will be
                plotted with the geometry superimposed and saved.
        '''
        # this datelist is updated after step 2 of the cropping procedure
        if len(flx_geom_gdf) != len(mask_params):
            logger.debug(f'{icos_site}: GEOM_GDF & MP EQUAL?!\n'
                         f'gdf: {flx_geom_gdf[["date", "PAR", "GPP_DT_VUT_50", "SW_IN_F"]]}\n mask_params:{mask_params}')
            raise ValueError('Nr. of obs. in flx_geom_gdf and mask_params is not equal.')
        
        datelist = flx_geom_gdf.loc[flx_geom_gdf.name == icos_site, 'startdate'].dt.date
        dtakes = flx_geom_gdf.loc[datelist.index, 'dataTakeID']
        datetimes = flx_geom_gdf.loc[datelist.index, 'startdate'].dt.strftime('%Y-%m-%d %H:%M')
        #if len(flx_geom_gdf.name.unique()) > 1: # crop DF to single site
        #    flx_geom_gdf = flx_geom_gdf[flx_geom_gdf.name == icos_site].reset_index(drop=True)
        
        if dimred == None:
            data_dir = self.img_dir
            nm_min = 400
            if sr == 'vis':
                nm_max = 700
            elif sr == 'vnir':
                nm_max = 1000
            elif sr == 'vswir':
                nm_max = 2500
            flx_geom_gdf['NIRvP'] = np.nan
        else:
            data_dir = self.img_dir / f'DR_{dimred}_imgs'
        
        img_paths = [list((data_dir).glob(f'*{dt}_{icos_site}_6km_crop.tif')) for dt in dtakes]
        
        epsg = self.flx_loc.loc[self.flx_loc.name == icos_site, 'sensorcrs'].item()
        crs_utm = proj.CRS.from_epsg(epsg)
        # Local UTM coordinates used for cropping, modeled geoms are in LAEA
        flx_loc = self.icos_db.loc[self.icos_db.name == icos_site,
                                   'geometry'].to_crs(crs_utm)
        flx_roi_plot = flx_loc.buffer(500)
        box_geom_plot = geometry.box(*flx_roi_plot.total_bounds)
        
        crs_lam = proj.CRS.from_epsg('3035')
        transf = proj.Transformer.from_crs(crs_lam, crs_utm, always_xy=True)
        
        cubes = [0]*len(datelist)
        wlss = [0]*len(datelist)
        itrans = [0]*len(datelist)
        bounds = [0]*len(datelist)
    
        for i,path in enumerate(img_paths):
            # imagery must exist and ID must be unique
            if len(path) == 0: # check if file is present
                raise ValueError(f'The cropped {self.sensor} image with dataTakeID {dtakes.iloc[i]} could not be found.')
            elif len(path) > 1:
                raise ValueError(f'dataTakeID {dtakes.iloc[i]} not unique. Please investigate.')
            with rio.open(path[0]) as src:
                cubes[i] = src.read()
                if dimred == None:
                    wlss[i] = [float(w) for w in list(src.descriptions)]
                itrans[i] = src.meta['transform'] # for plotting & cropping
                bounds[i] = src.bounds
                na_val = src.nodata
        cubes = [np.einsum('kli->lik', c) for c in cubes]
        exts = [riop.plotting_extent(c, itrans[i]) for i,c in enumerate(cubes)]
        
        if dimred == None:
            wl_min = _fnv(wlss[0], nm_min)
            wl_max = _fnv(wlss[0], nm_max) + 1
            nbands = len(wlss[0][wl_min:wl_max])
            logger.debug(f'initial band length: {len(wlss[0])}, sr_min + index: '
                         f'{nm_min}-{wl_min}, sr_max + index: {nm_max}-{wl_max}, reduced band length: {nbands}')

        cube_list = [0]*len(datelist)
        geom_px_avgs = [0]*len(datelist)
        if np.isnan(na_val):
            na_len = lambda x: len(x[np.isnan(x)])
        else:
            na_len = lambda x: len(x[x == na_val])
        
        for i,path in enumerate(img_paths):
            # NA handling not necessary anymore: Done at the end of self._model_geoms
            # Non-vegetation px masking for direct aggregation of HSI (DR case treated differently)
            if dimred == None:
                row = flx_geom_gdf.iloc[i]
                mp = mask_params.iloc[i]
                #logger.debug(f'{icos_site}_{dtakes.iloc[i]}: _mask_px args:\n'
                #             f'ix:{i}\n row: {row}\n mask_param:{mp}')
                veg_mask, cube_rgb = self._mask_px(mask_param=mp, cube=cubes[i], wls=wlss[i],
                                                   row=row, loc=flx_loc, ext=exts[i])
                for band in range(0, len(wlss[i])):
                    cubes[i][veg_mask, band] = np.nan
            
            lam_poly = flx_geom_gdf.loc[i, 'geometry']
            logger.debug(f'LAEA poly coords: {lam_poly.bounds}')
            utm_poly = stransform(transf.transform, lam_poly)
            logger.debug(f'UTM poly coords: {utm_poly.bounds}')
            logger.debug(f'UTM raster bounds: {bounds[i]}')
            geom_cube, otrans = _local_mask(cubes[i], itrans[i], [utm_poly],
                                            crop=True, all_touched=True, nodata=na_val)
            # TODO: values>1 should also be removed (total amount of pixels is very small)
            #describe(temp, axis=None)
            if (na_len(geom_cube) / geom_cube.size) > 0.05:
                logger.debug(f'{icos_site}_{dtakes.iloc[i]}: '
                             f'{round((len(geom_cube[geom_cube == na_val]) / geom_cube.size)*100, 2)}%'
                             ' of regular pixels have NaN values.')
            if (len(geom_cube[geom_cube < 0]) / geom_cube.size) > 0.05:
                logger.debug(f'{icos_site}_{dtakes.iloc[i]}: '
                             f'{round((len(geom_cube[geom_cube < 0]) / geom_cube.size)*100, 2)}%'
                             ' of regular pixels have < 0 values. Negative reflectance values are converted to NaN.')
            
            if not np.isnan(na_val):
                geom_cube[geom_cube == na_val] = np.nan
            if dimred == None: # might otherwise cause problems for DR with negative resulting coef values
                geom_cube[geom_cube < 0] = np.nan
                geom_cube = geom_cube[:, :, wl_min:wl_max] # Crop dimensions according to chosen spectral range
                if upw == True: # HSI are cropped to 400-700 nm and multiplied with PAR
                    par = flx_geom_gdf.loc[i, 'PAR']
                    geom_cube = geom_cube * par
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    nir = np.nanmean(geom_cube[:, :, _fnv(wlss[i], 800):
                                               _fnv(wlss[i], 850)+1], axis=2)
                    red = np.nanmean(geom_cube[:, :, _fnv(wlss[i], 600):
                                               _fnv(wlss[i], 650)+1], axis=2)
                ndvi = (nir - red) / (nir + red)
                # NIRvP = NIRv * PAR, band specs: Dechant et al. (2022) - NIRVP: A robust structural proxy for sun-induced chlorophyll fluorescence and photosynthesis across scales
                # in practive for PRISMA: RED=[601, 646], NIR=[796, 849]
                nirvp = (ndvi * nir) * flx_geom_gdf.loc[i, 'PAR']
                flx_geom_gdf.loc[i, 'NIRvP'] = np.nanmean(nirvp)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                geom_px_avgs[i] = np.nanmean(geom_cube, axis=(1, 0))
            if np.isnan(geom_px_avgs[i]).all():
                flx_geom_gdf.loc[i, 'clouds'] = 'hsi_na'
                # also update in img_db for consistency
                self.img_db.loc[(self.img_db.name == icos_site) &
                                (self.img_db.dataTakeID == dtakes.iloc[i]),
                                'clouds'] = 'hsi_na'
                logger.warning(f'{icos_site}_{dtakes.iloc[i]}: Only NA pixels within geometry of interest.\n')

            cube_list[i] = geom_cube
            
            if dimred != None:
                continue
            if not save_plot:
                continue
            
            # clip RGB to plotting extent
            cube_rgb_plot, otrans = _local_mask( #4 indices since 4th channel is used for NA masking
                cube_rgb, itrans[i], [box_geom_plot], crop=True, indexes=[1,2,3,4])
            out_ext = riop.plotting_extent(cube_rgb_plot, otrans)
            patch1 = mpatches.Patch(fc='none', ec='crimson', lw=2, label='Flux ROI')
            patch2 = mpatches.Patch(fc='none', ec='turquoise', lw=2, label='Validation ROI')
            
            fig, ax = plt.subplots(1,1, figsize=(12,12))
            ax2 = ax.twinx() # dummy axes for legend with zorder = 1
            ax2.axis('off')
            ax.imshow(cube_rgb_plot, extent=out_ext)
            ax.title.set_text(f'{icos_site} : 80% footprint estimate, {datetimes.iloc[i]}, DT {dtakes.iloc[i]}')
            flx_geom_gdf[flx_geom_gdf.index == i].to_crs(crs_utm).plot(
                ax=ax, ec='crimson', fc='none', zorder=10, linewidth=2)
            ax.scatter(flx_loc.x.item(), flx_loc.y.item(), marker='x',
                       s=80, c='crimson', label='EC tower')
            ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            ytx = ax.get_yticks()
            xtx = ax.get_xticks()
            ax.set_yticks([ytx[0], ytx[-1]], visible=True, rotation='vertical')
            ax.set_xticks([xtx[0], xtx[-1]], visible=True, rotation='horizontal')
            ax2.set_zorder(1)
            ax2.legend(handles=[patch1, patch2], loc='lower right', fontsize=14, # frame width variable is missing
                       facecolor='w', edgecolor='black', framealpha=1, borderpad=0.4)
            if flx_geom_gdf.loc[i, 'clouds'] == 'hsi_na':
                ax.text(0.4, 0.6, 'NA pixel', size='30', c='xkcd:neon pink', transform=ax.transAxes)
            fig.savefig(self.img_dir / f'{img_paths[i][0].stem[:-8]}rgb_{self.ptype["mask"]}.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
        # np arrays cannot be saved as a GPKG element. Therefore 1 column in GDF for each band.
        if dimred == None:
            sr_band_ix = [f'b{str(x).zfill(3)}' for x in range(1, nbands + 1)]
        else:
            sr_band_ix = [f'comp{str(x).zfill(2)}' for x in range(1, cubes[0].shape[2] + 1)]
        geom_avg_df = pd.DataFrame([
            pd.Series(vals, index=sr_band_ix) for vals in geom_px_avgs])
        logger.debug(f'single geom_px_avgs: {geom_px_avgs[-1]}, band_cols in geom_avg_df: {sr_band_ix}')
        logger.debug(f'cube shape after sr adjustment: {np.shape(geom_cube)}')
        flx_hsi_gdf = pd.concat([flx_geom_gdf, geom_avg_df], axis=1)
        
        return flx_hsi_gdf, cube_list

    def hsi_geom_crop(self, icos_list, mask_params, response, date = None, sr = 'vnir',
                      zonal = False, upw = False, aggr = 'na', save = False, save_plot = False):
        '''
        Crops hyperspectral imagery to flux footprints derived from the 30-min
        interval of EC measurements at ICOS flux towers during or before the DESIS
        data take.
        Footprints are calculated using level 2 data of ICOS class 1 & 2 stations
        using the flux footprint model by Kljun et al. (2015) [1].
        
        [1] Kljun, N. et al. (2015): A simple two‐dimensional parameterisation
        for Flux Footprint Prediction (FFP). Geosci. Model Dev., 8, 3695‐3713.
        DOI:10.5194/gmd‐8‐3695‐2015.
        
        Args:
            icos_list (string or list of strings): Abbreviation of the ICOS
                site(s) for which hyperspectral imagery will be evaluated.
            mask_params (pandas.DataFrame): Parameters for masking non-
                vegetation pixels. Must be determined beforehand by visual
                analysis. Length must equal that of flx_geom_gdf.
            response (string): Name of the relevant ICOS (productivity) variable.
                Used to remove NA occurrences in self._model_geoms.
            date (string, optional): A specific date (YYYY-MM-DD) can be passed to
                process a single image.
            sr (string): Abbreviation of the spectral range within which
                HSI will be used, either 'vis', 'vnir' or 'vswir' (the latter
                is only available for PRISMA data).
            zonal (bool, optional): If true, zonal statistics (buffer value
                depending on ecosystem) will be calculated instead of FFs.
            upw (bool, optional): If true, reflectance values will be multi-
                plied with PAR from ICOS sensor data resulting in upwelling
                radiation (UPW). Only possible for 400-700nm.
            aggr (string): Type of aggregation of daily ICOS values, either
                "na", "mean" or "sum".
            save (bool, optional): If true, footprint geometries are saved as
                GeoPackage.
            save_plot (bool, optional): If true, ICOS site surroundings will be
                mapped with the geometry superimposed and saved.
        '''
        if (upw == True) & (sr != 'vis'):
            logger.info('upw is true but sr is not "vis". Since UPW ' +
                        'calculation is only possible for 400-700nm, ' +
                        'sr will be set to "vis"')
            sr = 'vis'
        if isinstance(icos_list, str):
            icos_list = [icos_list]
                         
        t = 'upw' if upw == True else 'ref' # upwelling radiation or reflectance
        a = 'bg' if zonal == True else 'ff' # flux footprint
        self.ptype = {'rad': t, 'mask': a, 'sr': sr} # processing info
        
        # seletion of productivity variables to be extracted (percentiles, partitioning)
        fluxvars0 = ['GPP_DT_VUT_05', 'GPP_DT_VUT_50', 'GPP_DT_VUT_95', 'GPP_NT_VUT_05',
                    'GPP_NT_VUT_50', 'GPP_NT_VUT_95', 'NEE_VUT_05', 'NEE_VUT_50',
                    'NEE_VUT_95', 'NEE_VUT_50_QC', 'PPFD_IN', 'SW_IN_F', 'SW_IN_F_QC']
        # additional data for incoming SW rad.
        no_ppfd, no_rad = self._ppfd_check(icos_list)
        # update list to remove stations without PPFD from UPW calculation
        if len(no_rad) > 0:
            icos_list = [x for x in icos_list if not x in no_rad]
        # TODO: Remove or separate check for NAs in SW_IN_F?
        
        flx_hsi_gdf_l = [pd.DataFrame(columns=['a'], index=range(0))]*len(icos_list)
        flx_imgs_l = [0]*len(icos_list)
        ql_list = [0]*len(icos_list)
        nnal = [0]*len(icos_list)
        na_sites = []
        
        for i,site in enumerate(pbar := tqdm(icos_list)):
            pbar.set_description(f'Processing {site}')
            mp_subset = mask_params[mask_params.name == site].reset_index(drop=True)
            if date == None:
                datelist = self.img_db.loc[self.img_db.name == site, 'startdate'].dt.date
            else:
                try:
                    dt.datetime.strptime(date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError('Incorrect date format, should be YYYY-MM-DD')
                datelist = self.img_db.loc[(self.img_db.name == site) & \
                                                 (self.img_db.date == date), 'startdate'].dt.date
            dtakes = self.img_db.loc[datelist.index, 'dataTakeID']
            
            if self.sensor == 'DESIS':
                img_paths = [list((self.img_dir).glob(f'*{dt}*SPECTRAL*.tif'))\
                             for i,dt in enumerate(dtakes)]
                # Only 1 image is used for getting CRS info as all images at one ICOS location share the same CRS!
            elif self.sensor == 'PRISMA':
                img_paths = [list((self.img_dir).glob(f'*{dt}*6km_crop'))\
                             for i,dt in enumerate(dtakes)]
        
            ip_check = pd.Series([len(x) > 1 for x in img_paths])
            if ip_check.any():
                raise ValueError('More than 1 matching img for dataTakeID {}'\
                                 .format(dtakes.iloc[ip_check[ip_check == True].index]))
            
            # (1) Load FF model parameters
            icos_subset, ZM, fluxvars, missd = self._load_icos_subset(
                site, datelist, fluxvars0, aggr, no_ppfd, zonal)
            
            # (2) Footprint modeling and geometry generation
            flx_geom_gdf_subset, ql_list[i], nnal[i] = self._model_geoms(
                    site, datelist, icos_subset, response, ZM, fluxvars, missd, zonal, upw)
            logger.debug(f'{site}: CRS of gdf after GEOMETRY (2): {flx_geom_gdf_subset.crs}')
            
            if len(flx_geom_gdf_subset) == 0: # Skip if all obs. of a site have ICOS NA values
                na_sites.append(site)
                continue

            # (3) Crop of hyperspectral imagery to footprint
            # 'zonal' arg not needed as cropping is identical for FF & zonal
            flx_hsi_gdf_subset, flx_imgs = self._crop_data_2_geoms(
                site, flx_geom_gdf_subset, mp_subset, None, sr, upw, save_plot)
            logger.debug(f'{site}: CRS of gdf after CROPPING (3): {flx_hsi_gdf_subset.crs}')
            
            flx_hsi_gdf_l[i] = flx_hsi_gdf_subset
            flx_imgs_l[i] = flx_imgs
        if sum(ql_list) > 0:
            logger.info(f'Poor quality ICOS data flags occurred at {sum(ql_list)} '
                        'sites. Detailed info in log file.')
        if len(na_sites) > 0:
            for s in na_sites:
                icos_list.remove(s) # truncate lists
            flx_hsi_gdf_l = [x for x in flx_hsi_gdf_l if not x.empty]
            flx_imgs_l = [x for x in flx_imgs_l if x != 0]
        
        flx_hsi_gdf_c = pd.concat(flx_hsi_gdf_l).reset_index(drop=True)
        flx_hsi_gdf = flx_hsi_gdf_c[flx_hsi_gdf_c.clouds != 'hsi_na'].reset_index(drop=True)
        if len(icos_list) == 1:
            return flx_hsi_gdf, flx_imgs_l[0], icos_list
        flx_imgs_c = [y for x in flx_imgs_l for y in x]
        
        # missing data is counted in _model_geoms (ICOS NAs, removed in method) and _crop_data_2_geoms (img NAs, removed here!)
        self._nna_icos = sum(nnal)
        nna_all = len(self.img_db[self.img_db.clouds == 'hsi_na']) + self._nna_icos
        logger.info(f'{nna_all} images with missing ICOS or HSI pixel data were '
                    'removed from the final geodataframe ("self.flx_hsi_gdf").')
        logger.debug('NA obs: {}'.format(self.img_db.loc[self.img_db.clouds.isin(
            ['icos_na', 'hsi_na']), ['name', 'date', 'clouds']]))
        
        self.flx_hsi_gdf = flx_hsi_gdf_c
        self.flx_imgs = flx_imgs_c
        
        if save == True:
            fpath = self.out_dir / (f'HSI_gdf_{self.sensor}_{self.ptype["sr"]}_'
                f'{self.ptype["mask"]}_{self.ptype["rad"]}.gpkg')
            if fpath.exists():
                fpath.unlink() # delete to prevent creation of multilayer gpkg
            logger.info(f'Saving file: {fpath.name}')
            flx_hsi_gdf.to_file(fpath, driver='GPKG')
        no_ppi = len(flx_hsi_gdf[flx_hsi_gdf.ppi_file == ''])
        if no_ppi > 0:
            logger.warning(f'No S2 PPI images available for {no_ppi} data takes')
            logger.debug(f'No PPI data takes:\n{flx_hsi_gdf.loc[flx_hsi_gdf.ppi_file == "", "dataTakeID"]}')
        return flx_hsi_gdf, flx_imgs_c, icos_list
    
### DIMRED FUNCTIONS ##########################################################
    
    def hsi_gdf_prep(self, icos_list, response, upw = False, zonal = False):
        '''
        Loads ICOS data and generates geometries of interest for dimension
        reduction. In case of FFs, the geometries are calculated using level 2
        data of ICOS class 1 & 2 stations with the flux footprint model by
        Kljun et al. (2015) [1].
        
        Args:
            icos_list (string or list of strings): Abbreviation of the ICOS
                site(s) for which hyperspectral imagery will be evaluated.
            upw (bool, optional): If true, reflectance values will be multi-
                plied with PAR from ICOS sensor data resulting in upwelling
                radiation (UPW). Only possible for 400-700nm.
            zonal (bool, optional): If true, zonal statistics (buffer value
                depending on ecosystem) will be calculated instead of FFs.
        
        [1] Kljun, N. et al. (2015): A simple two‐dimensional parameterisation
        for Flux Footprint Prediction (FFP). Geosci. Model Dev., 8, 3695‐3713.
        DOI:10.5194/gmd‐8‐3695‐2015.
        '''
        if isinstance(icos_list, str):
            icos_list = [icos_list]
                         
        t = 'upw' if upw == True else 'ref'
        a = 'bg' if zonal == True else 'ff' # flux footprint
        self.ptype = {'rad': t, 'mask': a} # processing info
        
        # seletion of productivity variables to be extracted (percentiles, partitioning)
        fluxvars0 = ['GPP_DT_VUT_05', 'GPP_DT_VUT_50', 'GPP_DT_VUT_95', 'GPP_NT_VUT_05',
                    'GPP_NT_VUT_50', 'GPP_NT_VUT_95', 'NEE_VUT_05', 'NEE_VUT_50',
                    'NEE_VUT_95', 'NEE_VUT_50_QC', 'PPFD_IN', 'SW_IN_F', 'SW_IN_F_QC']
        # additional data for incoming SW rad.
        no_ppfd, no_rad = self._ppfd_check(icos_list)
        # update list to remove stations without PPFD from UPW calculation
        if len(no_rad) > 0:
            icos_list = [x for x in icos_list if not x in no_rad]
        
        flx_geom_gdf_l = [0]*len(icos_list)
        ql_list = [0]*len(icos_list)
        nnal = [0]*len(icos_list)
        na_sites = []
        
        for i,site in enumerate(pbar := tqdm(icos_list)):
            pbar.set_description(f'Processing {site}')
            datelist = self.img_db.loc[self.img_db.name == site, 'startdate'].dt.date

            # (1) Load FF model parameters
            icos_subset, ZM, fluxvars, missd = self._load_icos_subset(
                site, datelist, fluxvars0, 'na', no_ppfd, zonal) # hardcoded "no aggregation"
            
            # (2) Footprint modeling and geometry generation
            flx_geom_gdf, ql_list[i], nnal[i] = self._model_geoms(
                    site, datelist, icos_subset, response, ZM, fluxvars, missd, zonal)
            
            if len(flx_geom_gdf) == 0: # Skip if all obs. of a site have ICOS NA values
                na_sites.append(site)
            flx_geom_gdf_l[i] = flx_geom_gdf
            
            # LOOP END: step (3a) is performed for all dataTakes at once
        self._nna_icos = sum(nnal)
        if sum(ql_list) > 0:
            logger.info(f'Poor quality ICOS data flags occurred at {sum(ql_list)}'
                        ' sites. Detailed info in log file.')
        
        if len(na_sites) > 0:
            for s in na_sites:
                icos_list.remove(s) # truncate list, DF with len=0 will be skipped by pd.concat
        flx_geom_gdf = pd.concat(flx_geom_gdf_l).reset_index(drop=True)

        return flx_geom_gdf, icos_list
    
    
    def _cloud_mask(self, hsi, wl, T1, t2, t3, t4, T5, T6, T7, T8, sh_only = False):
        '''
        Index-based cloud (shadow) masking method from Zhai, H. el al (2018) [1].
        We used EO-1 Hyperion wavelengths as demonstrated in the paper:
        blue 488, green 569, red 671, NIR 854, SWIR1 1609, SWIR2 2194.
        
        [1] Zhai, H. et al. (2018): Cloud/shadow detection based on spectral
        indices for multi/hyperspectral optical remote sensing imagery. ISPRS
        J. Photogramm. Remote Sens., 144, 235-253. DOI:10.1016/j.isprsjprs.2018.07.006
        
        Args:
            hsi (numpy.ndarray): Hyperspectral input data cube.
            wl (list of ints): Central wavelengths of hyperspectral bands
            T1 (float): Selected from {0.01, 0.1, 1, 10, 100}
            t2 (float): Adjusting coefficient, selected from  {1/10, 1/9, 1/8, 1/7,
               1/6, 1/5, 1/4, 1/3, 1/2}
            t3 (float): Adjusting coefficient, selected from {1/4, 1/3, 1/2, 2/3, 3/4}
            t4 (float): Adjusting coefficient, selected from {1/2, 2/3, 3/4, 4/5, 5/6}
            T5, T6 (int): Size parameters that control height & width of cloud
                search window.
            T7, T8 (int): Kernel size parameter, selected from {3, 5, 7, 9, 11}
            sh_only (bool, optional): If true, the cloud mask will be replaced by
                an all-false array of the same XY extent as the data cube.
        '''
        m_bl = hsi[:,:, _fnv(wl, 488)]
        m_gr = hsi[:,:, _fnv(wl, 569)]
        m_re = hsi[:,:, _fnv(wl, 671)]
        m_ni = hsi[:,:, _fnv(wl, 854)]
        m_s1 = hsi[:,:, _fnv(wl, 1609)]
        m_s2 = hsi[:,:, _fnv(wl, 2194)]

        ci1 = (m_ni + 2*m_s1) / (m_bl + m_gr + m_re)
        ci2 = (m_bl + m_gr + m_re + m_ni + m_s1 + m_s2) / 6

        T2 = np.nanmean(ci2) + (t2 * (np.nanmax(ci2) - np.nanmean(ci2)))

        prelim_cloud_mask = np.float32(np.logical_and(np.abs(ci1 - 1) < T1, ci2 > T2))
        final_cloud_mask = medianBlur(prelim_cloud_mask, T7).astype(np.uint8)
        
        T3 = np.nanmin(m_ni) + (t3 * (np.nanmean(m_ni) - np.nanmin(m_ni)))
        T4 = np.nanmin(m_bl) + (t4 * (np.nanmean(m_bl) - np.nanmin(m_bl)))

        prelim_cloud_shadow_mask = np.float32(np.logical_and(m_ni < T3, m_bl < T4))
        
        spatial_search_kernel = np.ones((T5, T6), dtype = np.float32)
        non_pseudo_cloud_shadow_position_mask = filter2D(final_cloud_mask, -1, spatial_search_kernel) < 0

        refined_cloud_shadow_mask = prelim_cloud_shadow_mask * non_pseudo_cloud_shadow_position_mask
        final_cloud_shadow_mask = medianBlur(refined_cloud_shadow_mask, T8)
        final_cloud_shadow_mask = final_cloud_shadow_mask.astype(np.uint8)
        
        if sh_only == True:
            final_cloud_mask = np.full(hsi.shape[:2], False)
        
        return final_cloud_mask, prelim_cloud_shadow_mask
    
    def _mask_px(self, mask_param, cube, wls, row, loc, ext = None, plot = False, save_plot = False):
        
        if (plot == True) & (ext == None):
            raise ValueError('If a cloud mask plot should be generated, "ext" cant be None.')
        
        titles = ['RGB', 'All masks applied', 'Cloud mask',
                  f'Dark shadow mask (VIS < {mask_param["sh_thresh"]})',
                  f'NDVI mask (< {mask_param["ndvi_thresh"]})', 'Cloud shadow mask']
        rows, cols, b = np.shape(cube)
        cubeT = cube.reshape(-1,b)
        rgb = HSI2RGB(wls, cubeT, rows, cols, 50, 0.0002)
        
        logger.debug(f'{row["name"]}_{row["dataTakeID"]}: conversion to NA - '
                     f'{len(cube[cube==0])} == 0; {len(cube[cube<0])} < 0; {len(cube[cube>1])} > 1')
        cube[cube == 0] = np.nan # zeros have to be converted for np.min functions to work properly
        cube[cube < 0] = np.nan
        cube[cube > 1] = np.nan # All these NA cases will be covered by na_mask
            
        if mask_param['cm+csm']:
            cm, csm = self._cloud_mask(hsi=cube, wl=wls, T1=mask_param['T1'],
                                       t2=.1, t3=.5, t4=.75, T5=30, T6=30, T7=3, T8=3, sh_only=False)
        else:
            cm, csm = [np.full(cube.shape[:2], False), np.full(cube.shape[:2], False)]
        
        # Dark shadows are not recognized: Extra mask for very low VIS refl. pixels
        sh_mask = np.full((rows, cols), False)
        for r,c in np.ndindex(cube.shape[:2]):
            spec = cube[r,c, :35] # only visible bands
            spec[np.isnan(spec)] = 0
            sh_mask[r,c] = (spec < mask_param['sh_thresh']).all()
            
        blue_thresh = .05
        if mask_param['blue_cor'] > 0 and mask_param['blue_cor'] != 1:
            blue_thresh = mask_param['blue_cor']
        blue_mask = cube[:, :, 11] > blue_thresh
            
        # Hyperspectral NDVI: Haboudane et. al (2004)
        r1 = cube[:,:,_fnv(wls, 800)] - cube[:,:,_fnv(wls, 670)]
        r2 = cube[:,:,_fnv(wls, 800)] + cube[:,:,_fnv(wls, 670)]
        ndvi_mask = (r1/r2) < mask_param['ndvi_thresh']
        
        na_mask = np.isnan(cube).all(axis=2) # border pixels
        
        if mask_param['blue_cor'] == 0: #combine all masks
            veg_mask = np.logical_or.reduce((cm, csm, sh_mask, ndvi_mask, na_mask))
        else:
            veg_mask = np.logical_or.reduce((cm, csm, sh_mask, ndvi_mask, na_mask, blue_mask))
            titles[3] = f'Blue mask (WL@490nm > {blue_thresh})'
        
        # Set mask values as alpha values of RGB for plotting invalid pixels as white/transparent
        rgb2 = np.concatenate([rgb, ~veg_mask[:, :, np.newaxis]], axis=-1)
        
        if plot == True:
            fig, ax = plt.subplots(2,3,figsize=(25, 15))
            fig.suptitle(f'{row["name"]} on {row["date"]}', fontsize=20, y=.93)
            ax[0,0].imshow(rgb, extent=ext)
            ax[0,1].imshow(rgb2, extent=ext)
            ax[0,2].imshow(cm, extent=ext)
            if mask_param['blue_cor'] == 0:
                ax[1,0].imshow(sh_mask, extent=ext)
            else:
                ax[1,0].imshow(blue_mask, extent=ext)
            ax[1,1].imshow(ndvi_mask, extent=ext)
            ax[1,2].imshow(csm, extent=ext)
            for i,a in enumerate(ax.flat):
                a.title.set_text(titles[i])
                a.scatter(loc.x.item(), loc.y.item(), marker='x', s=80, c='crimson', label='EC tower')
            if save_plot == True:
                fig.savefig(self.img_dir / f'PRS_L2D_STD_{row["dataTakeID"]}_{row["name"]}_masks.png',
                            pad_inches=0, bbox_inches='tight', dpi=150)
            plt.close(fig)

        return veg_mask, rgb2
    
    def hsi_dimred_prep(self, flx_geom_gdf, mask_params, plot = False,
                        save_plot = False):
        '''
        To prepare for DR, cropped imagery (6km crops) is loaded and non-
        vegetation pixels are masked. Valid pixels are concatenated to a large
        array (pixel x bands) suitable for DR, e.g. with autoencoders. Array
        is saved as .hdf file while backtransform coords are saved as .json.
        
        Args:
            flx_geom_gdf (geopandas.GeoDataFrame): Flux measurements that match
                the observation times of the hyperspectral imagery including
                geometries for cropping after DR (zonal or FF).
            mask_params (pandas.DataFrame): Parameters for masking non-
                vegetation pixels. Must be determined beforehand by visual
                analysis. Length & order must equal that of flx_geom_gdf.
            plot (bool, optional): If true, plots of tower surroundings with
                applied masks will be generated.
            save_plot (bool, optional): If true, the various masks will be
                plotted with ICOS site surroundings.
        '''
        if len(flx_geom_gdf) != len(mask_params):
            raise ValueError('Nr. of obs. in flx_geom_gdf and mask_params is not equal.')
        # The following operation is the DR equivalent to (3a) from self.hsi_geom_crop
        # after (1) & (2) in self.hsi_gdf_prep (masking, but without the cropping part: DR first)
        
        valid_rows = {}
        valid_cols = {}
        cubes_tf_l = [0]*len(flx_geom_gdf)
        prev_site = ''
        
        for ix, row in flx_geom_gdf.iterrows(): # Note - order matters for backtransform!
            site = row['name']
            if site != prev_site:
                logger.info(f'{site}: Masking non-vegetation')
            prev_site = site
            
            img_name = f'PRS_L2D_STD_{row["dataTakeID"]}_{row["name"]}_6km_crop.tif'
            epsg = self.flx_loc.loc[self.flx_loc.name == row['name'], 'sensorcrs'].item()
            crs_utm = proj.CRS.from_epsg(epsg)
            # Local UTM coordinates used for cropping, modeled geoms are in LAEA
            flx_loc = self.flx_loc.loc[self.flx_loc.name == row['name'], 'geometry'].to_crs(crs_utm)
            
            with rio.open(self.img_dir / img_name) as src:
                cube = src.read()
                crow, ccol = src.index(flx_loc.x.item(), flx_loc.y.item())
                wls = [float(w) for w in list(src.descriptions)]
                itrans = src.meta['transform']
            # Last 4 bands are removed due to prevalent NA values
            cube = np.einsum('kli->lik', cube)[:,:, :-4]
            wls = wls[:-4]
            out_ext = riop.plotting_extent(cube, itrans)
        
            veg_mask, _ = self._mask_px(mask_param=mask_params.iloc[ix], cube=cube,
                                        wls=wls, row=row, loc=flx_loc, ext=out_ext,
                                        plot=plot, save_plot=save_plot)
            
            ## LOOKUPS
            #1 create giant samples x bands table of all obs.
            #2 export giant table & lookup indices of each obs.
            #3 perform DR of choice and export samples x comps table
            #4 load comps table, backtransform & plot/sample with FF/BG
            logger.info(f'DT{row["dataTakeID"]} (ix:{ix}): Filtering & transposing.')
            valid_ix = np.where(~veg_mask) # invert mask to get zeros (= valid pixels, cloud px are 1/True)
            valid_rows[ix] = valid_ix[0].tolist()
            valid_cols[ix] = valid_ix[1].tolist()
            # length of saved coord lists provides information on how to split transp. concat. HSI array for backtransform
            
            # 2D array of valid pixels is created (by row)
            cube_tf = cube[valid_rows[ix], valid_cols[ix], :]
            cube_tf_na = np.where(np.isnan(cube_tf))

            # Fill in NAs by averaging from neighboring bands
            for r, b in zip(cube_tf_na[0], cube_tf_na[1]):
                if b == len(wls)-1: # upper edge case
                    cube_tf[r, b] = cube_tf[r, b - 1] + round(randint(1, 100)*1e-5, 5) # avoid ties
                elif b == 0: # lower edge case
                    cube_tf[r, b] = cube_tf[r, b + 1] + round(randint(1, 100)*1e-5, 5)
                else: # nanmean ensures that all NAs are filled
                    cube_tf[r, b] = np.nanmean([cube_tf[r, b - 1], cube_tf[r, b + 1]]) + round(randint(1, 100)*1e-5, 5)
            cubes_tf_l[ix] = cube_tf

        cube_tf_all = np.vstack(cubes_tf_l)
        
        # Whole VSWIR spectrum is always saved
        h5f = h5py.File(self.out_dir / 'DR_hsi_transp_{}_{}_{}.h5'\
                   .format(self.sensor, self.ptype['mask'], self.ptype['rad']), 'w')
        h5f.create_dataset('ds1', data=cube_tf_all)
        h5f.close()
        f = open(self.out_dir / 'DR_hsi_valid_rows.json','w')
        json.dump(valid_rows, f, indent=4)
        f.close()
        f = open(self.out_dir / 'DR_hsi_valid_cols.json','w')
        json.dump(valid_cols, f, indent=4)
        f.close()
        
        return cube_tf_all
    
    
    def load_pre_dimred_db(self, fdir = None, zonal = False, upw = False):
        '''
        Load the cleaned data frame of ICOS variables, SPEI and PPI matching
        the observation times of available hyperspectral imagery. The 
        
        Args:
            fdir (str, optional): Directory with saved HSICOS data.
            zonal (bool, optional): The GeoPackage is based on zonal statistics
                instead of footprint estimates.
            upw (bool, optional): The GeoPackage contains upwelling radiation
                instead of reflectance values.
        '''
        if fdir == None:
            fdir = self.out_dir
        t = 'upw' if upw == True else 'ref' # upwelling radiation or reflectance
        a = 'bg' if zonal == True else 'ff' # flux footprint
        self.ptype = {'rad': t, 'mask': a} # processing info
        
        fname = 'pre_DR_hsi_PRISMA_covars.gpkg'
        logger.info(f'Loading file {fname}')
        flx_geom_gdf = gpd.read_file(fdir / fname, driver='GPKG')
        flx_geom_gdf['startdate'] = pd.to_datetime(flx_geom_gdf.startdate)
        mask_params = pd.read_csv(fdir / 'updated_mask_params.csv')
        
        return flx_geom_gdf, mask_params
        

    def _jsonkeys2int(self, x):
        '''
        Helper function to convert JSON string keys into Python integer keys
        when importing dicts from JSON.
        '''
        if isinstance(x, dict):
                return {int(k):v for k,v in x.items()}
        return x

    def dimred_backtransform(self, dr_file, flx_geom_gdf):
        '''
        Back-transform dimension-reduced 2D (non-spatial) hyperspectral data
        to map coordinates according to lookup JSON dicts generated with the
        'hsi_dimred_prep' method. NaN values are preserved.
        
        Args:
            dr_file (string): Name of the HDF5 file containing DR pixel values
                of all observations in flx_geom_gdf.
            flx_geom_gdf (geopandas.GeoDataFrame): Flux measurements including
                geometries for cropping after DR. Only used as iterator here.
        '''
        dr_fp = dr_file.split('_')
        dr = re.sub(r'\d+', '', dr_fp[1])
        with h5py.File(self.out_dir / dr_file, mode='r') as f:
            cube_tf_comp = f['comps'][:]
        with open(self.out_dir / 'DR_hsi_valid_cols.json', 'r') as f:
            valid_cols = json.load(f, object_hook=self._jsonkeys2int)
        with open(self.out_dir / 'DR_hsi_valid_rows.json', 'r') as f:
            valid_rows = json.load(f, object_hook=self._jsonkeys2int)

        dr_dir = self.img_dir / f'DR_{dr_fp[1]}_imgs'
        # since only most promising hyperparameter config is used, other parameters like nepoch or dist. meas. are omitted in dr_dir.
        dr_dir.mkdir(parents=True, exist_ok=True)
        add = 0
        ncomp = cube_tf_comp.shape[1]
        # datatake loop
        for ix, row in flx_geom_gdf.iterrows(): 
            img_name = f'PRS_L2D_STD_{row["dataTakeID"]}_{row["name"]}_6km_crop.tif'
            with rio.open(self.img_dir / img_name) as src:
                ometa = src.meta
            
            cube_dr = np.empty((ometa['height'], ometa['width'], ncomp))
            cube_dr[:] = np.nan
            # backtransform loop
            for i, (r, c) in enumerate(zip(valid_rows[ix], valid_cols[ix])):
                cube_dr[r, c, :] = cube_tf_comp[i+add, :]
            
            dr_img = f'PRS_{dr}_{row["dataTakeID"]}_{row["name"]}_6km_crop.tif'
            ometa['count'] = ncomp
            ometa['nodata'] = np.nan
            logger.info(f'{row["name"]}_{row["dataTakeID"]}: Saving TIF of dimred HSI data.')
            with rio.open(dr_dir / dr_img, 'w', **ometa) as dst:
                for k in range(0, ncomp):
                    dst.write_band(k+1, cube_dr[:,:,k])
                    dst.set_band_description(k+1, f'{dr} component {k+1}')
            add += len(valid_rows[ix]) # add += i+1 would also be possible
        return
    
    def dimred_geom_crop(self, dr_file, flx_geom_gdf, mask_params, upw = False, save = False):
        '''
        After dimension reduction and backtransform of hyperspectral pixel
        values, this method is used to crop latent component data around the
        ICOS tower location. The GOI chosen in the 'hsi_gdf_prep' method
        determines if FFs or BGs are used.
        
        Args:
            dr_file (string): Name of the HDF5 file containing DR pixel values
                of all observations in flx_geom_gdf.
            flx_geom_gdf (geopandas.GeoDataFrame): Flux measurements that match
                the observation times of the hyperspectral imagery including
                geometries for cropping after DR (zonal or FF). CRS is LAEA
                Europe (EPSG:3035).
            mask_params (pandas.DataFrame): Parameters for masking non-
                vegetation pixels. Must be determined beforehand by visual
                analysis. Length must equal that of flx_geom_gdf.
            upw (bool, optional): If true, reflectance values will be multi-
                plied with PAR from ICOS sensor data resulting in upwelling
                radiation (UPW). Only possible for 400-700nm.
            save (bool, optional): If true, footprint geometries are saved as
                GeoPackage.
        '''
        dr_fp = dr_file.split('_')
        icos_list = flx_geom_gdf.name.unique().tolist()
        flx_comp_gdf_l = [0]*len(icos_list)
        flx_imgs_l = [0]*len(icos_list)
        #with tqdm(total=flx_geom_gdf.shape[0]) as pbar:    
            #for ix, row in flx_geom_gdf.iterrows():
        for i,site in enumerate(pbar := tqdm(icos_list)):
            mp_subset = mask_params[mask_params.name == site].reset_index(drop=True)
            flx_geom_gdf_subset = flx_geom_gdf[flx_geom_gdf.name == site].reset_index(drop=True)
            pbar.set_description('Processing %s' % site)
            # (3b) Crop of hyperspectral imagery to footprint
            # VNIR hardcoded for now
            flx_comp_gdf_subset, flx_imgs = self._crop_data_2_geoms(
                site, flx_geom_gdf_subset, mp_subset, dr_fp[1], 'vnir', upw, False)
            nna = len(flx_comp_gdf_subset[flx_comp_gdf_subset.clouds == 'hsi_na'])
            if nna == len(flx_comp_gdf_subset):
                logger.warning(f'{site}: All imagery acquisition dates have no'
                               ' ICOS or hyperspectral data available. Station'
                               ' will be missing in output DF.')
            flx_comp_gdf_l[i] = flx_comp_gdf_subset
            flx_imgs_l[i] = flx_imgs
        
        flx_comp_gdf_c = pd.concat(flx_comp_gdf_l).reset_index(drop=True)
        flx_comp_gdf = flx_comp_gdf_c[flx_comp_gdf_c.clouds != 'hsi_na'].reset_index(drop=True)
        if len(icos_list) == 1:
            return flx_comp_gdf, flx_imgs_l[0]
        else:
            flx_imgs_c = [y for x in flx_imgs_l for y in x]
                
        # Sort out imagery outside the ICOS record period and where flux or hyperspectral data have NAs
        nna_all = len(self.img_db[self.img_db.clouds == 'hsi_na']) + self._nna_icos
        # self._nna_icos is not modified again as this method is used repeatedly
        # whereas dimred_geom_crop and hsi_geom_crop only need to be executed once
        logger.info(f'{nna_all} images with missing ICOS or HSI pixel data were '
                    'removed from the final geodataframe ("self.flx_hsi_gdf").')
        logger.debug('NA obs: {}'.format(self.img_db.loc[self.img_db.clouds.isin(
            ['icos_na', 'hsi_na']), ['name', 'date', 'clouds']]))

        self.flx_comp_gdf = flx_comp_gdf
        self.flx_imgs = flx_imgs_c
        
        if save == True:
            fpath = self.out_dir / (f'DR_{dr_fp[1]}_gdf_{self.sensor}_'
                                    f'{self.ptype["mask"]}_{self.ptype["rad"]}_covars.gpkg')
            if fpath.exists():
                fpath.unlink()
            logger.info(f'Saving file: {fpath.name}')
            flx_comp_gdf.to_file(fpath, driver='GPKG')

        return flx_comp_gdf, flx_imgs_c
    
    
    def load_dimred_db(self, dimred, fdir = None, zonal = False, upw = False):
        '''
        Load the final data frame of ICOS flux estimates and dimension-reduced
        hyperspectral data sampled and averaged at tower locations (FF or BG).
        
        Args:
            dimred (string, optional): Describes the DR method and number of
                components, e.g. 'AE04', 'SiVM10', etc.
            fdir (str, optional): Directory with saved HSICOS data.
            zonal (bool, optional): The GeoPackage is based on zonal statistics
                instead of footprint estimates.
            upw (bool, optional): The GeoPackage contains upwelling radiation
                instead of reflectance values.
        '''
        if fdir == None:
            fdir = self.out_dir
        t = 'upw' if upw == True else 'ref' # upwelling radiation or reflectance
        a = 'bg' if zonal == True else 'ff' # flux footprint
        self.ptype = {'rad': t, 'mask': a} # processing info
        
        fname = f'DR_{dimred}_gdf_{self.sensor}_{self.ptype["mask"]}_{self.ptype["rad"]}_covars.gpkg'
        logger.info(f'Loading file {fname}')
        self.flx_comp_gdf = gpd.read_file(fdir / fname, driver='GPKG')
        self.flx_comp_gdf['startdate'] = pd.to_datetime(self.flx_comp_gdf.startdate)
        
        return self.flx_comp_gdf
            
### INCLUDE COVARIATES ########################################################

    def icos_ppi_get(self, icos_list, dataset = 'VI', day_range = 12,
                     datelist = None, save = True):
        '''
        This function should only be used AFTER HSI have been cropped to GOI
        (i.e. flx_hsi_gdf exists) as a substantial amount of data will have to
        be downloaded to get phenology information (S2 PPI data can only be
        cropped to 10k x 10k raster tiles before downloading). This function
        downloads and matches PPI data with HSI acquisition dates, crops the
        tiles to a bounding box with 500m radius around ICOS sites and saves
        them for following computations.
        
        Args:
            icos_list (string or list of strings): Abbreviation of the ICOS
                site(s) for which hyperspectral imagery will be evaluated.
            dataset (string): Either 'ST' (seasonal trajectories) or 'VI'
                (vegetation indices). Both contain PPI data.
            day_range (int): time window around the HSI dates within which PPI
                files will be searched for.
            datelist (pd.Series): When processing a single station, a series of
                specific timestamps can be supplied for processing.
            save (bool, optional): If true, info about PPI imagery filenames
                matched to HSI is saved in the flux geopackage.
        '''
        
        if isinstance(icos_list, str):
            icos_list = [icos_list]
        
        c = Client(debug=False, quiet=True, progress=False, sleep_max=120)
        if dataset == 'ST':
            ds = 'EO:EEA:DAT:CLMS_HRVPP_ST'
        elif dataset == 'VI':
            ds = 'EO:EEA:DAT:CLMS_HRVPP_VI'

        for site in icos_list:
            if datelist == None:
                datetimes = self.img_db.loc[(self.img_db.name == site) &
                                            (self.img_db.ppi_file == ''), 'startdate']
                n_hsi = len(self.img_db[self.img_db.name == site])
                if len(datetimes) == 0:
                    logger.info('{}: PPI data for all HSI data takes '.format(site) +
                                 'have already been downloaded. Site will be skipped.')
                    continue
                elif len(datetimes) < n_hsi:
                    logger.info('{}: Downloading PPI data for {}/{} HSI data takes.'\
                                .format(site, len(datetimes), n_hsi))
                elif len(datetimes) == n_hsi:
                    logger.info('{}: Downloading PPI data for all HSI data takes'.format(site))
            elif isinstance(datelist, pd.Series):
                datetimes = datelist
            else:
                raise ValueError('datelist has to be submitted as pd.Series.')
            dtakes = self.img_db.loc[datetimes.index, 'dataTakeID']
            ppi_files = [''] * len(datetimes)
            odir = self.img_dir.parent / 'Copernicus/S2PPI'
            odir.mkdir(parents=True, exist_ok=True)
            
            epsg = self.flx_loc.loc[self.flx_loc.name == site, 'sensorcrs'].item()
            crs_utm = proj.CRS.from_epsg(epsg)
            utm_zone = 'T' + crs_utm.utm_zone[:2] # target CRS
            flx_loc = self.icos_db.loc[self.icos_db.name == site, 'geometry']
            
            ext = flx_loc.buffer(0.0001).bounds.squeeze(axis=0).round(4) # 0.0001 deg to get 1 tile only
            flx_loc = flx_loc.to_crs(crs_utm)
            flx_loc_check = flx_loc.buffer(250)
            box_geom_check = geometry.box(*flx_loc_check.total_bounds)
            
            logger.debug('ppi_files: {} \n'.format(ppi_files))
            for i, ts in enumerate(pbar := tqdm(datetimes)):
                try:
                    ppi_exp = next(odir.glob('*_{}_PPI_{}_{}.tif'.format(dataset, site, dtakes.iloc[i])))
                    if ppi_exp.is_file(): # this check is necessary, e.g. if processing is interupted but hsicos has not been restarted to check for existing PPI crops at init
                        logger.info('{}: PPI data for DT:{} are already available. Skipping DT.'\
                                    .format(site, dtakes.iloc[i]))
                        ppi_files[i] = ''
                        datetimes.iloc[i] = 0
                        self.img_db.loc[(self.img_db.name == site) & (self.img_db.startdate == ts),
                                        'ppi_file'] = ppi_exp.name
                        continue
                except StopIteration:
                    pass # continue processing only if PPI TIF is not found
                pbar.set_description('Processing %s - TS %s' % (site, ts))
                ts1 = (ts - pd.Timedelta(days=round(day_range/2))).isoformat()
                ts2 = (ts + pd.Timedelta(days=round(day_range/2))).isoformat()# alternative: .strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                for sat in ['S2A', 'S2B']: # sat loop as backup against NA files
                
                    na_rate = 1
                    ct = 0 # while loop counter
                    query = {'datasetId': ds,
                             'boundingBoxValues': [{'name': 'bbox',
                                                    'bbox': [*ext]}],
                             'dateRangeSelectValues': [{'name': 'temporal_interval',
                                                        'start': ts1,
                                                        'end': ts2}],
                             'stringChoiceValues': [{'name': 'productType',
                                                     'value': 'PPI'}]
                             }
                    if dataset == 'VI':
                        query['stringChoiceValues'].append({
                            'name': 'platformSerialIdentifier',
                            'value': sat})
                    matches = c.search(query)
                    
                    if (len(matches.results) == 0) & (sat == 'S2B'):
                        # loop (i.e. download) is skipped, ppi_files entry remains empty
                        logger.error('{}: no usable S2B PPI data for TS{} (day range: {}). Trying S2A data now.\n'\
                                     .format(site, ts, day_range))
                        continue # If no success with S2B, continue with S2A
                    elif (len(matches.results) == 0) & (sat == 'S2A'):
                        logger.error('{}: no usable S2A & S2B PPI data for TS{} (day range: {}).\n'\
                                     .format(site, ts, day_range))
                        datetimes.iloc[i] = 0 # no suitable imagery available flag
                        break # No S2A+S2B -> break sat loop and start next TS iteration
                        
                    matches_filenames = pd.Series([x['filename'] for x in matches.results])
                    matches_exist = [(odir / x).is_file() for x in matches_filenames]
                    if all(matches_exist):
                        pass # skip download if all files are already present
                    else:
                        try:
                            matches.download(download_dir=odir) # download all "candidates" to have alternatives for NA check
                        except requests.exceptions.HTTPError as err:
                            logger.error('encountered HTTP error. Reconnecting after 30 seconds.')
                            logger.error(err.response.status_code)
                            logger.error(err.response.text)
                            sleep(60)
                            try:
                                matches.download(download_dir=odir)
                            except requests.exceptions.HTTPError as err1:
                                print(type(err.response.status_code))
                                if err1.response.status_code == 403 or err1.response.status_code == '403':
                                    logger.error('encountered HTTP 403 FORBIDDEN. Reconnecting after 300 seconds.')
                                    sleep(360)
                                    matches.download(download_dir=odir)
                                else:
                                    sleep(60)
                                    matches.download(download_dir=odir)
                        
                    matches_dates = pd.to_datetime(pd.Series([x[3:18] for x in matches_filenames]))
                    matches_datediffs = abs(matches_dates - ts).sort_values().drop_duplicates()
                    #logger.debug('all date matches: \n{}'.format(matches_dates))
                    #logger.debug('date diff matches: \n{}'.format(matches_datediffs))
                    
                    while na_rate > .7: # while NA rate is too high, the loop will iterate
                    # through adjacent dates in ascending order until a usable PPI file is found
                        logger.info('count: {}'.format(ct))
                        ft_ix = matches_datediffs.index[ct]
                        ft_date = matches_dates[ft_ix]
                        # date matching
                        match_ix = matches_dates[matches_dates == ft_date].index
                        if len(match_ix) == 1:
                            ppi_files[i] = matches_filenames[match_ix].item()
                        else: # crs matching
                            logger.warning('{}: {} date matches, possibly due to CRS edge case. files: {}\n'\
                                           .format(site, len(match_ix), matches_filenames[match_ix]))
                            correct_crs_ix = matches_filenames[match_ix].str.contains(utm_zone)
                            if correct_crs_ix.sum() == 1: # only 1 true value
                                ppi_files[i] = matches_filenames[match_ix][correct_crs_ix].item()
                            else:
                                ppi_files[i] = matches_filenames[match_ix][correct_crs_ix.index[0]]
                                logger.warning('{}: still more imagery remaining after CRS matching. First match is kept.\n'.format(site))
                        # If necessary, reproject PPI raster to match flux tower UTM zone
                        utm_tif = ppi_files[i][23:26]
                        if utm_tif != utm_zone:
                            with rio.open(odir / ppi_files[i]) as src:
                                transform, width, height = rio.warp.calculate_default_transform(
                                    src.crs, crs_utm, src.width, src.height, *src.bounds)
                                kwargs = src.meta.copy()
                        
                                kwargs.update({
                                    'crs': crs_utm,
                                    'transform': transform,
                                    'width': width,
                                    'height': height})
                                trans_tif_name = ppi_files[i][:23] + utm_zone + ppi_files[i][26:]
                        
                                with rio.open(odir / trans_tif_name, 'w', **kwargs) as dst:
                                    rio.warp.reproject(
                                        source=rio.band(src, 1),
                                        destination=rio.band(dst, 1),
                                        src_transform=src.transform,
                                        src_crs=src.crs,
                                        dst_transform=transform,
                                        dst_crs=crs_utm,
                                        resampling=rio.enums.Resampling.nearest)
                            ppi_files[i] = trans_tif_name
                            
                        # Check matched file for NAs. When too many, loop will test next file (chronologically).
                        with rio.open(odir / ppi_files[i]) as src:
                            ppi, _ = riom.mask(src, shapes=[box_geom_check],
                                               crop=True, nodata=-32768, indexes=1)
                        na_rate = len(ppi[ppi == -32768]) / np.size(ppi)
                        ct += 1
                        # Last iteration:
                        if (ct == len(matches_datediffs) - 1) & (sat == 'S2B'):
                            logger.warning('{}: Matching S2B files for TS{} (day range: {}) are >70% NA. Trying S2A data now.\n'\
                                           .format(site, ts, day_range))
                            na_rate = -9999
                        elif (ct == len(matches_datediffs) - 1) & (sat == 'S2A'):
                            logger.warning('{}: All S2A+S2B files for TS{} (day range: {}) are >70% NA.\n'\
                                           .format(site, ts, day_range))
                            na_rate != -9999
                            ppi_files[i] = ''
                            datetimes.iloc[i] = 0
                            break
                    if na_rate != -9999:
                        logger.debug('{} - TS{}: Found suitable file {}.\n'.format(site, ts, ppi_files[i]))
                        break # second sat loop will only start when all files are NA, else break sat loop      
                # END of satellite loop
            # END of timestamp loop
            logger.debug('NAr: {}, count: {}, sat: {}, ts-iter: {}\n'.format(na_rate, ct, sat, i))
            logger.debug('datetimes: {} \n'.format(datetimes))

            datetimes = datetimes[datetimes != 0] # remove entries without match
            dtakes = self.img_db.loc[datetimes.index, 'dataTakeID'] # for UID of each PPI img
            ppi_files = [x for x in ppi_files if x] # removes empty strings
            if len(datetimes) != len(ppi_files):
                raise RuntimeError('Length of datetimes and ppi_files for site {} does not align.'.format(site))
    
            ppi_files_new = ['{}_{}_PPI_{}_{}.tif'.format(f[3:18], dataset, site, dtakes.iloc[i]) for i,f in enumerate(ppi_files)]
            for i, ts in enumerate(datetimes):
                self.img_db.loc[(self.img_db.name == site) & (self.img_db.startdate == ts),
                                'ppi_file'] = ppi_files_new[i]
            # crop & save PPI rasters
            flx_loc_pb = flx_loc.buffer(1000)
            box_geom = geometry.box(*flx_loc_pb.total_bounds)
            for i, f in enumerate(ppi_files):
                with rio.open(odir / f) as src:
                    ppi, itrans = riom.mask(src, shapes=[box_geom], crop=True, nodata=-32768)
                    ometa = src.meta
                    
                ometa['width'] =  ppi.shape[2]
                ometa['height'] =  ppi.shape[1]
                # from rio docs: "Note that the interpretation of the 3 axes is (bands, rows, columns)"
                ometa['transform'] =  itrans
                with rio.open(odir / ppi_files_new[i], 'w', **ometa) as dest:
                    dest.write(ppi)
        # After cropping, delete all original files (originals start with their dataset abbr.)
        # this happens after site loop so that images can be recycled for nearby sites
        rm_files = list(odir.glob(dataset + '*.tif'))
        logger.debug('deleting originals: {}'.format([f.name for f in rm_files]))
        for f in rm_files:
            f.unlink()
        '''PPI file names are not saved as the cropped TIFs are matched with
        respective HSI when the hsicos class is initalized.'''          
        
        return
    
    def hsi_add_spei_ppi(self, flx_geom_gdf, mask_params, dimred = True,
                         zonal = False, rm_missing = False, rm_sites = [], save = False):
        '''
        After SPEI/PPI data have been acquired and processed using the methods
        icos_ppi_get/icos_cds_get, icos_eobs_pet and the SPEI calculation in R,
        they can be matched with the final flux+HSI data frame generated by the
        hsi_geom_crop method. All remaining imagery will be matched with corresp.
        SPEI & PPI values.
        
        Args:
            flx_geom_gdf (geopandas.GeoDataFrame): Flux measurements that match
                the observation times of the hyperspectral imagery including
                geometries for cropping (zonal or FF).
            mask_params (pandas.DataFrame): Parameters for masking non-
                vegetation pixels. Must be determined beforehand by visual
                analysis. Length must equal that of flx_geom_gdf.
            zonal (bool, optional): If true, PPI files will be cropped with
                zonal geometries instead of FFs.
            rm_missing (bool, optional): If true, observations with missing
                SPEI/PPI value are removed from flx_hsi_gdf.
            rm_sites (list of strings, optional): A list of flux site abbrevi-
                ations that should be excluded from the dataset (e.g. because
                a certain ecosystem should be excluded from analyses). 
            save (bool, optional): If true, the resulting data frame is saved
                as GeoPackage.
        '''
        if len(flx_geom_gdf) != len(mask_params):
            raise ValueError('Nr. of obs. in flx_geom_gdf and mask_params is not equal.')
        icos_list = flx_geom_gdf.name.unique().tolist()
        spei_list = [0]*len(icos_list)
        fdir = self.img_dir.parent / 'Copernicus/S2PPI'
        dform = '%Y-%m-%d'
        for i,site in enumerate(icos_list):
            spei_list[i] = pd.read_csv(self.icos_dir / site / f'eobs_spei_{site}.csv',
                               parse_dates=['Date'], date_format=dform)\
                [['Date', f'SPEI365_{site}']]
        spei = reduce(lambda df1, df2: pd.merge(df1, df2, how='outer', on='Date'), spei_list)
        spei['date_str'] = spei.Date.dt.strftime('%Y-%m-%d')

        if 'SPEI_365' not in flx_geom_gdf.columns:
            flx_geom_gdf['SPEI_365'] = -9999
        if 'PPI' not in flx_geom_gdf.columns:
            flx_geom_gdf['PPI'] = -9999
        
        for site in icos_list:
            datelist = flx_geom_gdf.loc[flx_geom_gdf.name == site, 'date']
            dtakes = flx_geom_gdf.loc[flx_geom_gdf.name == site, 'dataTakeID']
            #site_spei = spei.loc[spei.date_str.isin(datelist), ['date_str', 'SPEI365_{}'.format(site)]]
            
            epsg = self.flx_loc.loc[self.flx_loc.name == site, 'sensorcrs'].item()
            crs_utm = proj.CRS.from_epsg(epsg)
            #flx_loc = self.icos_db.loc[self.icos_db.name == site, 'geometry'].to_crs(crs_utm)
            crs_lam = proj.CRS.from_epsg('3035')
            transf = proj.Transformer.from_crs(crs_lam, crs_utm, always_xy=True)

            for i, date in enumerate(datelist):
                lam_poly = flx_geom_gdf.loc[datelist.index[i], 'geometry']
                ppi_geom = stransform(transf.transform, lam_poly)
                fname = flx_geom_gdf.loc[(flx_geom_gdf.name == site) &
                                         (flx_geom_gdf.date == date), 'ppi_file'].item()
                if len(fname) == 0:
                    logger.error(f'{site} {dtakes.iloc[i]}: No matching ppi_file ({date}).')
                    ppi_val = -9999
                else:
                    fpath = fdir / fname
                    with rio.open(fpath) as src: # ppi rasters are cropped
                        ppi, otrans = riom.mask(src, shapes=[ppi_geom], crop=True, all_touched=True,
                                                nodata=-32768, indexes=1)
                    logger.debug(f'{site} DT{dtakes.iloc[i]}: Matching ppi_file loaded.')
                    ppi = ppi / 10000
                    ppi[ppi < 0] = np.nan
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        ppi_val = np.nanmean(ppi, axis=(1, 0)) # mean of cropped area for more robust ppi estimate
                spei_val = spei.loc[spei.date_str == date, f'SPEI365_{site}'].item()
                if ppi_val == 0:
                    logger.warning(f'{site} - TS{date}: Calculated PPI = 0')
                flx_geom_gdf.loc[(flx_geom_gdf.name == site) &
                                 (flx_geom_gdf.date == date), 'PPI'] = ppi_val
                #logger.debug('{} DT:{} saved PPI value: {}'.format(site, dtakes.iloc[i], flx_geom_gdf.loc[
                #    (flx_geom_gdf.name == site) & (flx_geom_gdf.date == date), 'PPI'].item()))
                flx_geom_gdf.loc[(flx_geom_gdf.name == site) &
                                 (flx_geom_gdf.date == date), 'SPEI_365'] = spei_val
                #logger.debug('{} DT:{} saved SPEI value: {}'.format(site, dtakes.iloc[i], flx_geom_gdf.loc[
                #    (flx_geom_gdf.name == site) & (flx_geom_gdf.date == date), 'SPEI_365'].item()))

        if (flx_geom_gdf.SPEI_365 == -9999).any():
            logger.warning('No SPEI value was found for some acquisition dates!')
            logger.debug('SPEI == -9999:\n {}'.format(
                flx_geom_gdf.loc[flx_geom_gdf.SPEI_365 == -9999, ['name', 'date', 'dataTakeID']]))
        if (flx_geom_gdf.PPI == -9999).any() | np.isnan(flx_geom_gdf.PPI).any():
            logger.warning('No PPI value was found for some acquisition dates!')
            nan_ix = flx_geom_gdf.loc[(flx_geom_gdf.PPI == -9999) | np.isnan(flx_geom_gdf.PPI),
                             ['name', 'date', 'dataTakeID', 'PPI']].index
            logger.debug(f'PPI is -9999 or NaN:\n {flx_geom_gdf.loc[nan_ix]}')
            if rm_missing == True & len(nan_ix) > 0:
                flx_geom_gdf = flx_geom_gdf.drop(index=nan_ix).reset_index(drop=True)
                mask_params = mask_params.drop(index=nan_ix).reset_index(drop=True)
                logger.info(f'{len(nan_ix)} obs. with NaN PPI values / '
                            'no PPI imagery available have been removed from flx_geom_gdf.')
        
        zero_ix = flx_geom_gdf.loc[flx_geom_gdf.PPI == 0, :].index
        if len(zero_ix) > 0:
            logger.info(f'{len(zero_ix)} obs. with PPI = 0 have been removed from flx_geom_gdf.')
            flx_geom_gdf = flx_geom_gdf.drop(index=zero_ix).reset_index(drop=True)
            mask_params = mask_params.drop(index=zero_ix).reset_index(drop=True)
        
        if len(rm_sites) > 0:
            rm_ix = flx_geom_gdf.loc[flx_geom_gdf.name.isin(rm_sites),:].index
            logger.info(f'{len(rm_ix)} obs. from the sites {rm_sites} have been removed as specified in the call to "hsi_add_spei_ppi".')
            flx_geom_gdf = flx_geom_gdf.drop(index=rm_ix).reset_index(drop=True)
            mask_params = mask_params.drop(index=rm_ix).reset_index(drop=True)
        
        # dtype correction
        flx_geom_gdf = flx_geom_gdf.astype(
            {'GPP_DT_VUT_05':np.float64, 'GPP_DT_VUT_50':np.float64,
             'GPP_DT_VUT_95':np.float64, 'GPP_NT_VUT_05':np.float64,
             'GPP_NT_VUT_50':np.float64, 'GPP_NT_VUT_95':np.float64,
             'NEE_VUT_05':np.float64, 'NEE_VUT_50':np.float64,
             'NEE_VUT_95':np.float64, 'NEE_VUT_50_QC':np.int64,
             'SW_IN_F':np.float64, 'SW_IN_F_QC':np.int64, 'PAR':np.float64 })
        
        if save == True:
            if dimred == True:
                fpath = self.out_dir / f'pre_DR_hsi_{self.sensor}_covars.gpkg'
            else:
                fpath = self.out_dir / (f'HSI_gdf_{self.sensor}_{self.ptype["sr"]}_'
                    f'{self.ptype["mask"]}_{self.ptype["rad"]}_covars.gpkg')
            if fpath.exists():
                fpath.unlink()
            logger.info(f'Saving file {fpath.name}')
            flx_geom_gdf.to_file(fpath, driver='GPKG')
            mask_params.to_csv(self.out_dir / 'updated_mask_params.csv', index=False)
        
        return flx_geom_gdf, mask_params

    def load_hsi_db(self, fdir = None, sr = 'vnir', zonal = False, upw = False):
        '''
        Load the processed, cleaned version of the imagery data base encompas-
        sing ICOS flux estimates, spectrometer data cropped to ROIs, and co-
        variates (PPI, SPEI).
        
        Args:
            fdir (str, optional): Directory with saved HSICOS data.
            sr (string): The spectral range for which the GeoPackage
                includes HSI data.
            zonal (bool, optional): The GeoPackage is based on zonal statistics
                instead of footprint estimates.
            upw (bool, optional): The GeoPackage contains upwelling radiation
                instead of reflectance values.
            aggr (string, optional): Type of aggregation of daily ICOS values,
                either "na", "mean" or "sum".
        '''
        if (upw == True) & (sr != 'vis'):
            logger.info('upw is true but sr is not "vis". Since UPW ' +
                        'calculation is only possible for 400-700nm, ' +
                        'sr will be set to "vis"')
            sr = 'vis'
        if fdir == None:
            fdir = self.out_dir
        t = 'upw' if upw == True else 'ref' # upwelling radiation or reflectance
        a = 'bg' if zonal == True else 'ff' # flux footprint
        self.ptype = {'rad': t, 'mask': a, 'sr': sr} # processing info
        
        fname = (f'HSI_gdf_{self.sensor}_{self.ptype["sr"]}_{self.ptype["mask"]}'
                 f'_{self.ptype["rad"]}_covars.gpkg')
        logger.info(f'Loading file {fname}')
        self.flx_hsi_gdf = gpd.read_file(fdir / fname, driver='GPKG')
        self.flx_hsi_gdf['startdate'] = pd.to_datetime(self.flx_hsi_gdf.startdate)

        return self.flx_hsi_gdf
