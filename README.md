## drgpp - Data-driven productivity modeling with hyperspectral data and advanced dimension reduction methods

### Description
------------
This project contains files for reproducing results from the PhD thesis chapter "Sophisticated non-linear dimension reduction methods do not sur-
pass simpler approaches to explaining ecosystem productivity", in: "Improving hyperspectral monitoring of ecosystem functioning with novel latent variable transformations"

Preprocessing of hyperspectral data starts in 01\_hsi\_gpp\_preproc\_main.py, which is the main step-by-step processing file including methods for data acquisition from different data sources (e.g. Copernicus S2 PPI data or E-OBS meteorological data). PRISMA imagery as well as (ICOS) flux data cannot be downloaded via an API so that the user is required to obtain these datasets beforehand and put them in the respective folders "/data/fluxes/SITE_NAME>" and "/data/PRISMA". See the documentations of methods in "/fmch/fmch/hsicos.py" for detailed descriptions.

Dimension reduction with the unsupervised methods PCA, SiVM (simplex volume maximisation) and AE (autoencoder) is performed in 02\_hsi\_gpp\_dimred\_main.py. These analyses were conducted in a separate Python 3.7 environment using tensorflow 1.x because of restrictions in the deephyp module. A virtual environment with suitable module versions is required to use this file.

Calculation of the SPEI drought index was done in R and can be reproduced with the file 03\_hsi\_gpp\_spei\_main.R

Statistical analysis was also performed with R and can be reproduced with the notebook file 04\_hsi\_gpp\_stats\_main.ipynb if a jupyter R kernel is used. This file also works standalone without the preprocessing files if the input geopackages in the "/out" folder are used.

Please note that the code has been designed with Python 3.11.4 (except for dimension reduction) and R 4.3.0 on Linux Mint 21.1. Error handling exists but is not exhaustive.

### Requirements
------------

Python scripts require the following modules:

 * fmch (not available on conda/pypi, use included version, i.e. add folder structure to your Python PATH or add the functions and class in hsicos.py manually to your Python environment.)
 * numpy
 * xarray
 * pandas
 * geopandas
 * shapely
 * pyproj
 * rasterio
 * fiona
 * h5py
 * spectral
 * cv2
 * matplotlib
 * mpl_toolkits
 * pathlib
 * datetime
 * tqdm
 * pyeto
 * cdsapi
 * hda
 * logging
recommended:
 * HSI2RGB (https://github.com/JakobSig/HSI2RGB)
with Python 3.7:
 * pymf (not available on conda/pypi, use included version, source: https://github.com/cthurau/pymf)
 * tensorflow 1.14
 * deephyp 0.1.5
 * scikit-learn

R scripts require the following packages:

 * tidyverse
 * tidync
 * ggthemes
 * patchwork
 * repr
 * scales
 * viridis
 * data.table
 * mlr3verse
 * mlr3pipelines
 * mlr3extralearners
 * paradox
 * visNetwork
 * DALEX
 * DALEXtra
 * iml
 * SHAPforxgboost
 * ks
 * zoo
 * sf
 * hdf5r
 * lubridate
 * future
 * rstudioapi (for current file location)

### License for Flux footprint modeling software (ffp.py)
------------
Copyright (c) 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, Natascha Kljun

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

### License for PyMF matrix factorisation module: BSD 3-clause
------------
Copyright (c) 2014 Christian Thurau

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

### License for all other included Python & R scripts and modules: BSD 3-Clause
------------
Copyright (c) 2023, Floris Hermanns

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

