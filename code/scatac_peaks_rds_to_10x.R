library(SummarizedExperiment)
library(DropletUtils)

print('starting processing scATAC peaks data')
scATAC_all <- readRDS("data/scATAC-All-Hematopoiesis-MPAL-191120.rds")
col_all <- colData(scATAC_all)
scATAC_healthy <- readRDS("data/scATAC-Healthy-Hematopoiesis-191120.rds")
col_healthy <- colData(scATAC_healthy)
col_all[rownames(col_healthy), "ProjectClassification"] <- col_healthy$BioClassification
colData(scATAC_all) <- col_all
DropletUtils::write10xCounts(x = assay(scATAC_all), path = "data/scATAC/peaks")
write.csv(rowData(scATAC_all), "data/scATAC/peaks/var.csv")
write.csv(colData(scATAC_all), "data/scATAC/peaks/obs.csv")
print('saved scATAC-seq peaks data in data/scATAC/peaks/')
