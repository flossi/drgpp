## =============================================================================
##
## misc_functions.R
##
## The 'load_flx_gdf' function in this script is used for loading .gpkg files
## resulting from preprocessing in prisma_ggp_preproc_main.py. The .gpkgs contain
## combined information about estimated GPP at different ICOS sites (and metadata),
## associated cropped and averaged hyperspectral bands as well as PPI (derived from
## Copernicus S2 data) and SPEI (derived from E-OBS data).
## Furthermore the file contains adapted plotting functions for SHAP values and
## a PLS learner for mlr3.
## 
## Author: Floris Hermanns
##
## Date Created: 2024-01-04
##
## =============================================================================

library(paradox)
library(R6)
library(mlr3)

load_flx_gdf <- function(wdir, sensor, sr, mask, rad, aggr = 'na',
                         response = 'GPP_DT_VUT_50', dimred = 'none', return_df = FALSE,
                         exclude_es = 'none', site_only = NULL, year_only = NULL) {
  #' @title Load ICOS + HSI geopackage
  #'
  #' @description Load a geopackage file of preprocessed & cropped hyperspectral
  #' information combined with time-matched flux tower data for analyses like
  #' dimension reduction followed by a (supervised) statistical learning approach.
  #' 
  #' @param wdir directory with saved HSICOS data.
  #' @param sensor type of hyperspectral sensor, either "PRISMA" or "DESIS".
  #' @param sr spectral range of the input data, either "vnir" (visible & near
  #' infrared) or "vswir" (visible, near- & shortwave infrared).
  #' @param mask type of spatial statistic, either "ff" (flux footprint) or
  #' "bg" (buffer geometry).
  #' @param rad type of spectral data, either "upw" (upwelling radiation) or
  #' "ref" (reflectance).
  #' @param aggr The type of value aggregation that has been performed. Either
  #' "na" (no aggr.), "sum" (daily sum) or "mean" (daily daytime mean).
  #' @param response The response variable to be included in the output data.frame.
  #' Can be any ecosystem productivity variable measured at ICOS sites.
  #' @param dimred If DR has been applied, dimred describes the DR method used
  #' and number of components, e.g. 'AE04', 'SiVM10', etc. Required to choose
  #' specific .gpkg file of DR imagery.
  #' @param exclude_es Vector of IGBP classes to exclude. By default, no eco-
  #' systems are excluded.
  #' @param return_df If true, a regular data frame instead of a sf spatial data
  #' frame will be returned.
  #' @param site_only Use the ICOS abbreviation of a specific site as argument
  #' to return only observations from this site.
  #' @param year_only Only return observations from the selected year.
  #' 
  #' @return a list containing the gap-filled, cleaned geopackage as a data.frame
  #' and the number of hyperspectral bands.
  #' 
  
  fname <- ifelse(dimred == 'none',
                  sprintf('HSI_gdf_%s_%s_%s_%s_covars.gpkg', sensor, sr, mask, rad),
                  sprintf('DR_%s_gdf_%s_%s_%s_covars.gpkg', dimred, sensor, mask, rad))
  #fname <- sprintf('all_sites_%s_%s_%s_%s_covars.gpkg', sensor, sr, mask, rad)
  fpath <- file.path(wdir, fname)
  
  flx_db <- st_transform(st_read('/home/flossi/Work/fluxes/flx_sites.gpkg'), 3035)
  flx_db <- cbind(flx_db, st_coordinates(flx_db))
  
  flx_hsi_gdf <- st_read(fpath)
  flx_hsi_gdf$ecosystem <- 0
  # import ecosystem type from flx_sites file
  for (site in unique(flx_hsi_gdf$name)) {
    flx_hsi_gdf[flx_hsi_gdf$name == site, 'ecosystem'] <- flx_db[flx_db$name == site, 'ecosystem', drop=TRUE]
    flx_hsi_gdf[flx_hsi_gdf$name == site, 'X'] <- flx_db[flx_db$name == site, 'X', drop=TRUE]
    flx_hsi_gdf[flx_hsi_gdf$name == site, 'Y'] <- flx_db[flx_db$name == site, 'Y', drop=TRUE]
  }
  if (all(exclude_es != 'none')) {
    flx_hsi_gdf = flx_hsi_gdf[!flx_hsi_gdf$ecosystem %in% exclude_es, ]
  }
  
  ppi_z_ix = which(flx_hsi_gdf$PPI == 0)
  if (length(ppi_z_ix) != 0L) {
    zero_sites = flx_hsi_gdf[ppi_z_ix, 'name', drop=TRUE]
    message(sprintf('rows %s (%s) contain zero PPI values and will be removed.',
                    paste(sprintf('%s', ppi_z_ix), collapse=','),
                    paste(sprintf('%s', zero_sites), collapse=',')))
    flx_hsi_gdf <- flx_hsi_gdf[-ppi_z_ix, ]
  }
  gpp_z_ix = which(flx_hsi_gdf[, response, drop=TRUE] == 0)
  if (length(gpp_z_ix) != 0L) {
    zero_sites = flx_hsi_gdf[gpp_z_ix, 'name', drop=TRUE]
    message(sprintf('rows %s (%s) contain zero productivity values and will be removed.',
                    paste(sprintf('%s', gpp_z_ix), collapse=','),
                    paste(sprintf('%s', zero_sites), collapse=',')))
    flx_hsi_gdf <- flx_hsi_gdf[-gpp_z_ix, ]
  }
  rownames(flx_hsi_gdf) <- NULL # reset index
  
  cn <- colnames(flx_hsi_gdf)
  last_band <- ifelse(dimred == 'none',
                      max(cn[grepl('b', cn)]), max(cn[grepl('comp', cn)])) # max returns alphanumerically last value
  nbands <- ifelse(dimred == 'none', as.integer(substr(last_band, 2, 4)),
                   as.integer(substr(last_band, 5, 6)))
  
  # check for NAs in hyperspectral data
  if (dimred == 'none') {
    X <- flx_hsi_gdf[, sprintf('b%03d', seq(1:nbands)), drop=TRUE]
    na_cols <- unique(which(is.na(X), arr.ind = T)[ ,2])
    if (length(na_cols) != 0) {
      message(sprintf('%s: bands %s contain NAs. Will be gapfilled with mean from adjacent bands.',
                      fname, paste(sprintf('%s', na_cols), collapse=',')))
      for (col in na_cols) {
        na_ix <- which(is.na(X[col]))
        if (length(na_ix) > 1) {
          for (ix in na_ix) {
            X[ix, col] <- mean(X[ix-1, col], X[ix+1, col])
          }
        } else if (length(na_ix) == 1) {
          X[na_ix, col] <- mean(X[na_ix-1, col], X[na_ix+1, col])
        }
      }
    }
    which9999 <- apply(X, 1, function(x) length(which(x==-9999)))
    ix9999 <- which(which9999 > 0)
    if (length(ix9999) > 0) {
      message(sprintf('%s: rows %s contain all -9999s. Observations will be removed.',
                      fname, paste(sprintf('%s', ix9999), collapse=',')))
    }
    which0 <- apply(X, 1, function(x) length(which(x==0)))
    ix0 <- which(which0 > 0)
    if (length(ix0) > 0) {
      message(sprintf('%s: rows %s contains all zeros. Observations will be removed.',
                      fname, paste(sprintf('%s', ix0), collapse=',')))
    }
    X <- cbind(X, NIRvP=flx_hsi_gdf$NIRvP) # added here as column is currently not present in DR gpkgs
  } else {
    X <- flx_hsi_gdf[, sprintf('comp%02d', seq(1:nbands)), drop=TRUE]
  }

  # combine gap-filled bands with relevant columns from GPKG
  flx_hsi_gdf_clean <- cbind(flx_hsi_gdf[, c(response, 'name', 'date', 'PPI',
                                             'SPEI_365', 'ecosystem', 'X', 'Y')], X)
  flx_hsi_gdf_clean <- flx_hsi_gdf_clean %>% mutate_if(is.character, as.factor)
  
  # add ecosystem type dummy variables
  #flx_hsi_gdf_clean$forest <- ifelse(grepl('MF|DBF|ENF|EBF', flx_hsi_gdf_clean$ecosystem), 1, 0)
  dummies <- model.matrix(~0+flx_hsi_gdf_clean[, 'ecosystem', drop=TRUE])
  dummynames <- paste0('es.', lapply(levels(as.factor(flx_hsi_gdf$ecosystem)), tolower))
  colnames(dummies) <- dummynames
  flx_hsi_gdf_f <- cbind(flx_hsi_gdf_clean, dummies)
  
  if (!is.null(site_only)) {
    flx_hsi_gdf_f = flx_hsi_gdf_f[flx_hsi_gdf_f$name == site_only, ]
  }
  
  if (!is.null(year_only)) {
    if (is.integer(year_only)) {
      year_only <- as.character(year_only)
    }
    flx_hsi_gdf_f = flx_hsi_gdf_f[grepl(year_only, flx_hsi_gdf_f$date, fixed = T), ]
    # remove all ecosystem columns with all zeros
    dropcols <- c()
    for (name in dummynames) {
      print(paste(name, 'is a dummy!'))
      if (all(flx_hsi_gdf_f[, name, drop=T] == 0)) {
        print(paste(name, 'will be removed!'))
        dropcols <- c(dropcols, name)
      }
    }
    print(dropcols)
    flx_hsi_gdf_f <- flx_hsi_gdf_f[, !(names(flx_hsi_gdf_f) %in% dropcols)]
  }
  
  if (dimred == 'none') {
    if (length(ix9999) > 0 || length(ix0) > 0) {
      flx_hsi_gdf_f <- flx_hsi_gdf_f[-union(ix9999, ix0), ]
    }
  }

  if (return_df == TRUE) {
    flx_hsi_gdf_f = flx_hsi_gdf_f[,,drop=TRUE][,-which(names(flx_hsi_gdf_f) %in% c('geom'))]
  } else {
    # TEMP: could be replaced by using coords from flx_db
    flx_hsi_gdf_f$geom <- st_centroid(flx_hsi_gdf_f) %>% 
      st_geometry()
  }
  
  return(list('hsi_df' = flx_hsi_gdf_f, 'nbands' = nbands))
}




label.feature <- function(x){
  # a saved list of some feature names that I am using
  labs <- SHAPforxgboost::labels_within_package
  # but if you supply your own `new_labels`, it will print your feature names
  # must provide a list.
  if (!is.null(new_labels)) {
    if(!is.list(new_labels)) {
      message("new_labels should be a list, for example,`list(var0 = 'VariableA')`.\n")
    }  else {
      message("Plot will use your user-defined labels.\n")
      labs = new_labels
    }
  }
  out <- rep(NA, length(x))
  for (i in 1:length(x)){
    if (is.null(labs[[ x[i] ]])){
      out[i] <- x[i]
    }else{
      out[i] <- labs[[ x[i] ]]
    }
  }
  return(out)
}

shap.plot.summary2 <- function(data_long, x_bound = NULL, dilute = FALSE,
                               scientific = FALSE, my_format = NULL){
  
  if (scientific){label_format = "%.1e"} else {label_format = "%.3f"}
  if (!is.null(my_format)) label_format <- my_format
  # check number of observations
  N_features <- data.table::setDT(data_long)[,uniqueN(variable)]
  if (is.null(dilute)) dilute = FALSE
  
  nrow_X <- nrow(data_long)/N_features # n per feature
  if (dilute!=0){
    # if nrow_X <= 10, no dilute happens
    dilute <- ceiling(min(nrow_X/10, abs(as.numeric(dilute)))) # not allowed to dilute to fewer than 10 obs/feature
    set.seed(1234)
    data_long <- data_long[sample(nrow(data_long),
                                  min(nrow(data_long)/dilute, nrow(data_long)/2))] # dilute
  }
  
  x_bound <- if (is.null(x_bound)) max(abs(data_long$value))*1.1 else as.numeric(abs(x_bound))
  plot1 <- ggplot(data = data_long) +
    coord_flip(ylim = c(-x_bound, x_bound)) +
    geom_hline(yintercept = 0) + # the y-axis beneath
    # sina plot:
    ggforce::geom_sina(aes(x = variable, y = value, color = stdfvalue),
                       method = "counts", maxwidth = 0.7, alpha = 0.7) +
    # print the mean absolute value:
    geom_text(data = unique(data_long[, c("variable", "mean_value")]),
              aes(x = variable, y=-Inf, label = sprintf(label_format, mean_value)),
              size = 5, alpha = 0.7,
              hjust = -0.2,
              fontface = "bold") + # bold
    # # add a "SHAP" bar notation
    # annotate("text", x = -Inf, y = -Inf, vjust = -0.2, hjust = 0, size = 3,
    #          label = expression(group("|", bar(SHAP), "|"))) +
    scale_color_gradient(low="#FFCC33", high="#6600CC",
                         breaks=c(0,1), labels=c(" Low","High "),
                         guide = guide_colorbar(barwidth = 12, barheight = 0.3)) +
    theme_bw() +
    theme(axis.line.y = element_blank(),
          axis.ticks.y = element_blank(), # remove axis line
          legend.position="bottom",
          legend.title=element_text(size=16),
          legend.text=element_text(size=14),
          axis.title.x= element_text(size = 16)) +
    # reverse the order of features, from high to low
    # also relabel the feature using `label.feature`
    scale_x_discrete(limits = rev(levels(data_long$variable)),
                     labels = label.feature(rev(levels(data_long$variable))))+
    labs(y = "SHAP value (impact on model output)", x = "", color = "Feature value  ")
  return(plot1)
}


LearnerRegrPls <- R6Class("LearnerRegrPls", inherit = LearnerRegr,
                          public = list(
                            #' @description
                            #' Creates a new instance of this [R6][R6::R6Class] class.
                            initialize = function() {
                              ps = ps(scale = p_lgl(tags = "train"), center = p_lgl(tags = "train"), ncomp = p_int(tags = c("train", "predict")))
                              ps$values = list()
                              super$initialize(
                                id = "regr.pls",
                                packages = "pls",
                                feature_types = c("numeric"),
                                predict_types = c("response"),
                                param_set = ps,
                                properties = NULL
                              )
                              }
                            ),
                          
                          private = list(
                            .train = function(task) {
                              # get parameters for training
                              pars = self$param_set$get_values(tags = "train")
                              # set column names to ensure consistency in fit and predict
                              self$state$feature_names = task$feature_names
                              formula = task$formula()
                              data = task$data()
                              # use the mlr3misc::invoke function (it's similar to do.call())
                              mlr3misc::invoke(pls::plsr,
                                               formula = formula,
                                               data = data,
                                               .args = pars)
                              },
                            
                           .predict = function(task) {
                             # get parameters with tag "predict"
                             pars = self$param_set$get_values(tags = "predict")
                             # get newdata and ensure same ordering in train and predict
                             newdata = task$data(cols = self$state$feature_names)
                             pred = mlr3misc::invoke(predict, self$model, newdata = newdata, .args = pars)
                             list(response = pred[,1,1])
                             }
                           )
                          )

shap.sum.facet <- function(data_long, namedict, x_bound = NULL, my_format = NULL,
                           min_color_bound = "#FFCC33", max_color_bound = "#6600CC") {
  label_format = "%.2f"
  namix <- as.character(str_rank(levels(data_long$variable)))
  print(paste(c("Orig labels: ", levels(data_long$variable)), collapse=" "))
  print(paste(c("Alph. rank: ", namix), collapse=" "))
  print(paste(c("Cor. labels: ", parse(text = namedict[namix])), collapse=" "))
  if (!is.null(my_format)) {
    label_format <- my_format}
  N_features <- data.table::setDT(data_long)[, data.table::uniqueN(variable)]
  nrow_X <- nrow(data_long)/N_features
  x_bound <- if (is.null(x_bound)) {c(min(data_long$value) * 1.1, max(data_long$value) * 1.1)} else {as.numeric(x_bound)}
  plot1 <- ggplot(data = data_long) +
    coord_flip(ylim = c(x_bound[1], x_bound[2])) + geom_hline(yintercept = 0) +
    geom_text(data = unique(data_long[, c("variable", "mean_value", "dr_mod")]),
              aes(x = variable, y = -Inf, label = sprintf(label_format, mean_value)),
              size = 3, alpha = 0.7, hjust = -0.2, fontface = "bold", check_overlap = TRUE) +
    ggforce::geom_sina(aes(x = variable, y = value, color = stdfvalue), method = "counts", maxwidth = 0.7, alpha = 0.6) +
    #scale_x_discrete(limits = rev(levels(data_long$variable)),
    #             labels = label.feature(rev(levels(data_long$variable)))) + 
    scale_x_discrete(limits=rev(levels(data_long$variable)), labels=parse(text = rev(namedict[namix]))) +
    scale_colour_viridis(breaks = c(0, 1), labels = c(" Low", "High "),
                         guide = guide_colourbar(barwidth = 12, barheight = 0.3)) +
    #scale_color_gradient(low = min_color_bound, high = max_color_bound, breaks = c(0, 1),
    #                     labels = c(" Low", "High "), guide = guide_colorbar(barwidth = 12, barheight = 0.3)) +
    theme_bw() + theme(axis.line.y = element_blank(),
                       axis.ticks.y = element_blank(),
                       legend.position = "bottom",
                       legend.title = element_text(size = 10),
                       legend.text = element_text(size = 8),
                       axis.title.x = element_text(size = 10),
                       axis.text = element_text(size=10),
                       panel.background = element_rect(fill='grey83'),
                       panel.grid.major=element_line(colour='grey79'),
                       panel.grid.minor=element_line(colour='grey79'),
                       legend.key.width = unit(dev.size()[1] / 2, 'cm'),
                       legend.key.height = unit(dev.size()[1] / 10, 'cm')) +
    labs(y = "SHAP value (impact on model output)", x = "", color = "Predictor value  ") +
    ggplot2::facet_wrap(~ dr_mod, nrow = 1)
  return(plot1)
}

# Aux function for shap.dep.facet from SHAPforxgboost pkg
plot.label <- function(plot1, show_feature){
  if (show_feature == 'dayint'){
    plot1 <- plot1 +
      scale_x_date(date_breaks = "3 years", date_labels = "%Y")
  } else if (show_feature == 'AOT_Uncertainty' | show_feature == 'DevM_P1km'){
    plot1 <- plot1 +
      scale_x_continuous(labels = function(x)paste0(x*100, "%"))
  } else if (show_feature == 'RelAZ'){
    plot1 <- plot1 +
      scale_x_continuous(breaks = c((0:4)*45), limits = c(0,180))
  }
  plot1
}

shap.dep.facet <- function(data_long, x, y = NULL, color_feature = NULL, data_int = NULL,  # if supply, will plot SHAP
                           dilute = FALSE, smooth = TRUE, size0 = NULL, add_hist = FALSE, add_stat_cor = FALSE, alpha = NULL, jitter_height = 0, jitter_width = 0, ...) {
  if (is.null(y)) y <- x
  data0 <- data_long[variable == y, .(variable, value)] # the shap value to plot for dependence plot
  data0$x_feature <- data_long[variable == x, rfvalue]
  data0$dr_mod <- data_long[variable == x, dr_mod]

  # Note: strongest_interaction can return NULL if there is no color feature available
  # Thus, we keep this condition outside the next condition
  if (!is.null(color_feature) && color_feature == "auto") {
    color_feature <- strongest_interaction(X0 = data0, Xlong = data_long)
  }
  if (!is.null(color_feature)) {
    data0$color_value <- data_long[variable == color_feature, rfvalue]
  }
  if (!is.null(data_int)) data0$int_value <- data_int[, x, y]

  nrow_X <- nrow(data0)
  if (is.null(dilute)) dilute = FALSE
  if (dilute != 0){
    dilute <- ceiling(min(nrow(data0)/10, abs(as.numeric(dilute))))
    # not allowed to dilute to fewer than 10 obs/feature
    set.seed(1234)
    data0 <- data0[sample(nrow(data0), min(nrow(data0)/dilute, nrow(data0)/2))] # dilute
  }

  # for dayint, reformat date
  if (x == "dayint"){
    data0[, x_feature:= as.Date(data0[, x_feature], format = "%Y-%m-%d",
                                origin = "1970-01-01")]
  }

  if (is.null(size0)) {
    size0 <- if (nrow(data0) < 1000L) 1 else 0.4
  }

  if (is.null(alpha)) {
    alpha <- if (nrow(data0) < 1000L) 1 else 0.6
  }
  plot1 <- ggplot(
    data = data0,
    aes(x = x_feature,
        y = if (is.null(data_int)) value else int_value,
        color = if (!is.null(color_feature)) color_value else NULL)
    ) +
    geom_jitter(
      size = size0,
      width = jitter_width,
      height = jitter_height,
      alpha = alpha,
      ...
    ) +
    labs(y = if (is.null(data_int)) paste0("SHAP value for ", label.feature(y)) else
      paste0("SHAP interaction values for\n", label.feature(x), " and ", label.feature(y)),
         x = label.feature(x),
         color = if (!is.null(color_feature))
           paste0(label.feature(color_feature), "\n","(Feature value)") else NULL) +
    scale_color_gradient(low="#FFCC33", high="#6600CC",
                         guide = guide_colorbar(barwidth = 10, barheight = 0.3)) +
    theme_bw() +
    theme(legend.position = "bottom",
          legend.title = element_text(size = 10),
          legend.text = element_text(size = 8)) 

  # a loess smoothing line:
  if (smooth) {
    plot1 <- plot1 +
      geom_smooth(method = "loess", color = "red", size = 0.4, se = FALSE)
  }
  plot1 <- plot.label(plot1, show_feature = x)

  # add correlation
  if (add_stat_cor) {
    plot1 <- plot1 + ggpubr::stat_cor(method = "pearson")
  }

  # add histogram
  if (add_hist) {
    plot1 <- ggExtra::ggMarginal(
      plot1, type = "histogram", bins = 50, size = 10, color = "white"
    )
  }

  plot1 <- plot1 + ggplot2::facet_wrap(~ dr_mod, nrow = 1)
}


fig <- function(width = 7, heigth = 7, pointsize = 12, res = 120) {
 options(repr.plot.width = width, repr.plot.height = heigth, repr.plot.pointsize = pointsize, repr.plot.res = res)
}

"%||%" <- function(a, b) {
  if (!is.null(a)) a else b
}

geom_flat_violin <- function(mapping = NULL, data = NULL, stat = "ydensity",
                        position = "dodge", trim = TRUE, scale = "area",
                        show.legend = NA, inherit.aes = TRUE, ...) {
  layer(
    data = data,
    mapping = mapping,
    stat = stat,
    geom = GeomFlatViolin,
    position = position,
    show.legend = show.legend,
    inherit.aes = inherit.aes,
    params = list(
      trim = trim,
      scale = scale,
      ...
    )
  )
}

GeomFlatViolin <-
  ggproto(
    "GeomFlatViolin",
    Geom,
    setup_data = function(data, params) {
      data$width <- data$width %||%
        params$width %||% (resolution(data$x, FALSE) * 0.9)

      # ymin, ymax, xmin, and xmax define the bounding rectangle for each group
      data %>%
        dplyr::group_by(.data = ., group) %>%
        dplyr::mutate(
          .data = .,
          ymin = min(y),
          ymax = max(y),
          xmin = x,
          xmax = x + width / 2
        )
    },

    draw_group = function(data, panel_scales, coord)
    {
      # Find the points for the line to go all the way around
      data <- base::transform(data,
                              xminv = x,
                              xmaxv = x + violinwidth * (xmax - x))

      # Make sure it's sorted properly to draw the outline
      newdata <-
        base::rbind(
          dplyr::arrange(.data = base::transform(data, x = xminv), y),
          dplyr::arrange(.data = base::transform(data, x = xmaxv), -y)
        )

      # Close the polygon: set first and last point the same
      # Needed for coord_polar and such
      newdata <- rbind(newdata, newdata[1,])

      ggplot2:::ggname("geom_flat_violin",
                       GeomPolygon$draw_panel(newdata, panel_scales, coord))
    },

    draw_key = draw_key_polygon,

    default_aes = ggplot2::aes(
      weight = 1,
      colour = "grey20",
      fill = "white",
      size = 0.5,
      alpha = NA,
      linetype = "solid"
    ),

    required_aes = c("x", "y")
  )

