library(tibble)
library(RColorBrewer)
library(dynutils)
library(stringr)
library(Hmisc)
library(plyr)


source("code/knit_table.R") # You will need to have in the same folder knit_table.R and this plotSingleAtlas.R

# parameters: 
# - 'csv_file_path' would be the path of the csv file 


plotSingleAtlas <- function(csv_file_path, outdir="outputs/metrics_plots", sep=';'){
  
  metrics_tab_lab <- read.csv(csv_file_path, sep=sep)
  
  # get metrics names from columns
  metrics <- colnames(metrics_tab_lab)[!colnames(metrics_tab_lab) %in% c("data", "rqr", "method")]
  metrics <- gsub("\\.", "/", metrics)
  metrics <- gsub("_", " ", metrics)
  metrics <- plyr::mapvalues(metrics, from = c("ASW label", "ASW label/batch", "graph conn"), 
                             to = c("Cell type ASW", "Batch ASW","graph connectivity"))
  # metrics names as they are supposed to be ordered
  group_batch <- c("PCR batch", "Batch ASW", "graph connectivity")
  group_bio <- c("NMI cluster/label", "ARI cluster/label", "Cell type ASW", 
                 "isolated label silhouette")
  group_time <- c("reference time", "query time" )
  # set original values of number of metrics
  n_metrics_batch_original <- sum(group_batch %in% metrics)
  n_metrics_bio_original <- sum(group_bio %in% metrics)
  n_metrics_time_original <- sum(group_time %in% metrics)
  
  
  # order metrics present in the table
  matching.order <- match(c(group_batch, group_bio), metrics)
  metrics.ord <- metrics[matching.order[!is.na(matching.order)]]
  
  # data scenarios to be saved in file name
  data.scenarios <- unique(as.character(metrics_tab_lab$data))
  
  
  
  
  ###### Plot one figure for each data scenario
  for (dt.sc in data.scenarios){
  
    ind.scen <- grep(dt.sc, metrics_tab_lab$data)
    metrics_tab_sub <- metrics_tab_lab[ind.scen, ]
    
    # info on ratio
    #ratio <- metrics_tab_sub$rqr
    
    methods <- as.character(metrics_tab_sub$method)
    # methods_name <- capitalize(methods)
    methods_name <- plyr::mapvalues(methods, 
                                    from = c("multigrate","Seurat", "Seuratrpca", "mnnCorrect", "Bbknn", "trVAE", "scvi", "Liger", "Combat", "Saucie", "Fastmnn", "Desc", "scanvi", "Scgen"), 
                                    to = c("Multigrate", "Seurat v3 CCA", "Seurat v3 RPCA", "MNN", "BBKNN", "trVAE", "scVI", "LIGER", "ComBat", "SAUCIE", "fastMNN", "DESC", "scANVI", "scGen*"))
    
    
    ##### Create dataframe 
    metrics_tab <- as.data.frame(metrics_tab_sub[, !colnames(metrics_tab_sub) %in% c("data", "rqr", "method")])
    metrics_tab[metrics_tab == ""] <- NA
    colnames(metrics_tab) <- metrics
    
    #add Methods column
    metrics_tab <- add_column(metrics_tab, "Method" = methods_name, .before = 1)
    
    ## Remove columns that are full NAs
    na.col <- apply(metrics_tab, 2, function(x) sum(is.na(x)) == nrow(metrics_tab))
    
    # redefine numbers of metrics per group
    if(sum(colnames(metrics_tab)[na.col] %in%  group_batch) > 0){
      n_metrics_batch <- n_metrics_batch_original - sum(colnames(metrics_tab)[na.col] %in%  group_batch)
    } else {
      n_metrics_batch <- n_metrics_batch_original
    }
    
    
    if(sum(colnames(metrics_tab)[na.col] %in% group_bio) > 0){
      n_metrics_bio <- n_metrics_bio_original - sum(colnames(metrics_tab)[na.col] %in% group_bio)
    } else{
      n_metrics_bio <- n_metrics_bio_original
    }
    
    metrics_tab <- metrics_tab[, !na.col]
    
    
    ## Scores should be already scaled [0,1] - however, we aim to compute the scores based on the min-max scaled metrics
    scaled_metrics_tab <- as.matrix(metrics_tab[, -1])
    scaled_metrics_tab <- apply(scaled_metrics_tab, 2, function(x) scale_minmax(x))
    
    # calculate average score by group and overall
    score_group1 <- rowMeans(scaled_metrics_tab[, 1:n_metrics_batch], na.rm = T)
    score_group2 <- rowMeans(scaled_metrics_tab[, (1+n_metrics_batch):(ncol(scaled_metrics_tab)-n_metrics_time_original)], 
                             na.rm = T)
    
    score_all <- (0.4*score_group1 + 0.6*score_group2)
    
    scaled_metrics_tab <- as.data.frame(scaled_metrics_tab)
    scaled_metrics_tab <- add_column(scaled_metrics_tab, "Method" = metrics_tab$Method, .before = 1)
    metrics_tab <- scaled_metrics_tab
    metrics_tab <- add_column(metrics_tab, "Overall Score" = score_all, .after = "Method")
    metrics_tab <- add_column(metrics_tab, "Batch Correction" = score_group1, .after = "Overall Score")
    metrics_tab <- add_column(metrics_tab, "Bio conservation" = score_group2, .after = 3+n_metrics_batch)
    
    # metrics_tab <- add_column(metrics_tab, "Ratio" = ratio, .after = "Method")
    
    # order methods by the overall score
    metrics_tab <- metrics_tab[order(metrics_tab$`Overall Score`,  decreasing = T), ]
    write.csv(metrics_tab, file = paste0(outdir, "/", dt.sc, "_summary_scores.csv"), quote = F)
    
    
    # Delete rows that are empty
    rowsNA <- which(is.na(metrics_tab$`Overall Score`))
    if(length(rowsNA) >0){
      metrics_tab <- metrics_tab[-rowsNA, ]
    }
    
    
    # Defining column_info, row_info and palettes
    row_info <- data.frame(id = metrics_tab$Method)
   
    column_info <- data.frame(id = colnames(metrics_tab),
                              group = c("Text", "Score overall", 
                                        rep("Removal of batch effects", (1 + n_metrics_batch)),
                                        rep("Cell type label variance", (1 + n_metrics_bio)),
                                        rep("Time", n_metrics_time_original)), 
                              geom = c("text", "bar", "bar",
                                       rep("circle", n_metrics_batch), "bar", rep("circle", n_metrics_bio), 
                                       rep("bar", n_metrics_time_original)),
                              width = c(7,2,2, rep(1,n_metrics_batch), 2, rep(1,n_metrics_bio), rep(2, n_metrics_time_original)),
                              overlay = F)
    
    # defining colors palette
    palettes <- list("Score overall" = "YlGnBu",
                     "Removal of batch effects" = "BuPu",
                     "Cell type label variance" = "RdPu",
                     "Time" = "YlOrRd")
    

    g <- scIB_knit_table(data = metrics_tab, column_info = column_info, row_info = row_info, palettes = palettes, usability = F, atac = T)  
    now <- Sys.time()
    ggsave(paste0(outdir, "/", format(now, "%Y%m%d_%H%M%S_"), dt.sc, "_summary_metrics.pdf"), g, device = cairo_pdf, width = 297, height = 420, units = "mm")
    ggsave(paste0(outdir, "/", format(now, "%Y%m%d_%H%M%S_"), dt.sc, "_summary_metrics.tiff"), g, device = "tiff", dpi = "retina", width = 297, height = 420, units = "mm")
    ggsave(paste0(outdir, "/", format(now, "%Y%m%d_%H%M%S_"), dt.sc, "_summary_metrics.png"), g, device = "png", dpi = "retina", width = 210, height = 297, units = "mm")
    
  }
  
  
  
}