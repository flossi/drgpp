## =============================================================================
##
## spei_functions.R
##
## Functions for calculating non-parametric SPEI values from E-OBS precipitation
##and temperature data.
##
## Author: Floris Hermanns
##
## Date Created: 2023-03-17
##
## =============================================================================

icos_spei_prep <- function(fluxdir, icos_site, start_date = as.Date('1950-01-01')) {
  df_nc <- tidync(sprintf(file.path(fluxdir, '%s/eobs_spei_%s.csv'), icos_site, icos_site)) %>% 
    hyper_tibble()
  
  df_nc$Date <- start_date + df_nc$time
  
  # Calculate hydrolog. balance from precipitation and PET
  df_nc$bal = df_nc$pre - df_nc$pet
  return(df_nc)
}

kde_spei <- function(fluxdir, df, icos_site, k_width=365, overwrite=F, save=T) {
  speifile <- sprintf(file.path(fluxdir, '%s/eobs_spei_%s.csv'), icos_site, icos_site)
  if (file.exists(speifile) && overwrite == F) {
    print(sprintf('%s: SPEI csv already exists. Skipping.', icos_site))
    return(NULL)
  }
  
  start_date <- df[1, 'Date', drop=TRUE]
  start_doy <- yday(start_date)
  start <- year(start_date)
  
  end_date <- df[nrow(df), 'Date', drop=TRUE]
  end_doy <- yday(end_date)
  end <- year(end_date) # assuming daily data
  ny <- end - start + 1
  #seq_months <- append(seq(1,12,1), c(18,24))
  #agg <- append(c(1,3,5,7,10,14,21), 
  #              round((365/12) * seq_months, digits = 0))
  
  ### -----------------------
  ### EPDF by KDE
  ### ----------------------
  
  idx <- with(df,which(format(Date,'%m-%d')=='02-29'))
  n <- 365 # DOY (constant)
  # Remove all 29th Feb. occurences from data
  df_nly <- within(df, bal <- replace(bal, idx-1, mean(bal[idx+(-1:0)])))[-idx, ] %>%
    select(Date, pre, pet, bal) %>%
    mutate(Doy = c(seq(start_doy, n), rep(seq(1, n), ny-2), seq(1, end_doy)))
  
  dummy <- data.frame(Year = seq(start,end, by = 1))
  # NAs are ignored due to the 365-value window width
  df_nly[,paste0("BAL_",n)] <- zoo::rollapply(df_nly$bal, k_width, FUN=mean,
                                         fill = NA, align = "right", na.rm=T)
  
  #KDE for each Doy
  for(j in 1:n){
    #get vector of targeted variable for j Day of Year
    x <- df_nly %>% # Does not include NAs before start_doy or after end_doy
      dplyr::filter(Doy == j) %>%
      pull(paste0("BAL_",n))
    x2 <- x[!is.na(x)]
    p_c <- rep(-999., length(x)) # correction vector for insertion
    
    # calculate bandwidth
    f_kde <- kde(x2, h = hlscv(x2))
    # get probabilities
    p_kde <- pkde(x2, f_kde)
    
    # add values back into correct positions (including NAs)
    p_c[!is.na(x)] <- p_kde
    p_c[is.na(x)] <- NA
    p <- p_c
    if (any(p == -999., na.rm = T)){
      stop('There are less values in the KDE probability vector than expected.')
    }
    # Add NA value to DOY vectors before start_doy and after end_doy for full year records
    if (j < start_doy) {
      p <- c(NA, p)
    } else if (j > end_doy) {
      p <- c(p, NA)
    }
    
    
    for(k in 1:ny) {
      C0 <- 2.515517
      C1 <- 0.802853
      C2 <- 0.010328
      d1 <- 1.432788
      d2 <- 0.189269
      d3 <- 0.001308
      
      if(is.na(p[[k]]) == T) {
        dummy[k,paste0(j)] <- NA
      } else if(p[[k]] <= 0.5) {
        w <-  sqrt(log(1 / p[[k]]^2 ))
        dummy[k,paste0(j)] <- (-1) * (w - ( (C0 + C1 * w + C2 * w^2) / (1 + d1 * w + d2*w^2 + d3*w^3) ) )
      } else if (p[[k]] > 0.5) {
        w <-  sqrt(log(1 / (1 - p[[k]])^2 ))
        dummy[k,paste0(j)] <- w - ( (C0 + C1 * w + C2 * w^2) / (1 + d1 * w + d2*w^2 + d3*w^3) ) 
      }
    }
  }

  dummy_l <- dummy %>%
    pivot_longer(-Year)
  dummy_l$name <- as.integer(dummy_l$name)
  # remove NA entries before start_doy and after end_doy
  dummy_l_c <- dummy_l[!((dummy_l$Year == end & dummy_l$name > end_doy) |
                           (dummy_l$Year == start & dummy_l$name < start_doy)), ]
  
  df_nly[paste0('SPEI',n,'_',icos_site)] <- dummy_l_c$value
  if (save == T) {
    write_csv(df_nly, speifile)
    print(sprintf('SPEI data have been computed and saved as .csv for ICOS site %s (%s - %s).',
                  icos_site, start_date, end_date))
  }
  else {
    print(sprintf('SPEI data have been computed for ICOS site %s (%s - %s).',
                  icos_site, start_date, end_date))
  }

  return(df_nly)
}
