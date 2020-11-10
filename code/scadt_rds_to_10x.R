library(SummarizedExperiment)
library(DropletUtils)

print('starting processing scADT data')
scADT_all <- readRDS("data/granja2019/raw/scADT-All-Hematopoiesis-MPAL-191120.rds")
col_all <- colData(scADT_all)
scADT_healthy <- readRDS("data/granja2019/raw/scADT-All-Hematopoiesis-MPAL-191120.rds")
col_healthy <- colData(scADT_healthy)
col_all[rownames(col_healthy), "ProjectClassification"] <- col_healthy$BioClassification
colData(scADT_all) <- col_all
scADT_all@assays@data@listData[["counts"]] = as(scADT_all@assays@data@listData[["counts"]], "sparseMatrix")  
DropletUtils::write10xCounts(x = assay(scADT_all), path = "data/granja2019/scADT")
write.csv(rowData(scRNA_all), "data/granja2019/scADT/var.csv", row.names = F)
write.csv(colData(scRNA_all), "data/granja2019/scADT/obs.csv")
print('saved scADT-seq data in data/granja2019/scADT')
