## =============================================================================
##
## icos_spei.R
##
## This script is used for calculating non-parametric SPEI values from E-OBS
## precipitation and temperature data.
##
## Author: Floris Hermanns
##
## Date Created: 2023-03-17
##
## =============================================================================

rm(list=ls())
library(tidync)
library(tidyverse)
library(lubridate)
library(ggthemes)
library(ks)
library(zoo)
getCurrentFileLocation <-  function()
{
  this_file <- commandArgs() %>% 
    tibble::enframe(name = NULL) %>%
    tidyr::separate(col=value, into=c("key", "value"), sep="=", fill='right') %>%
    dplyr::filter(key == "--file") %>%
    dplyr::pull(value)
  if (length(this_file)==0)
  {
    this_file <- rstudioapi::getSourceEditorContext()$path
  }
  return(dirname(this_file))
}
wdir0 <- getCurrentFileLocation()
source(file.path(wdir0, 'R/spei_functions.R'))
fluxdir <- file.path(wdir0, 'data/fluxes')

# SPEI calculation =============================================================

flx_prisma <- data.table::fread(fluxdir, select = c('name'))
icos_list <- unique(flx_prisma$name)

for (site in icos_list) {
  print(site)
  df_nc <- icos_spei_prep(fluxdir, site)
  df_spei <- kde_spei(fluxdir, df_nc, site, save=T)
}
