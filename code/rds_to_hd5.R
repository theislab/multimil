library(HDF5Array)
library(SummarizedExperiment)

scRNA_all <- readRDS("data/scRNA-All-Hematopoiesis-MPAL-191120.rds")
col_all <- colData(scRNA_all)
scRNA_healthy <- readRDS("data/scRNA-Healthy-Hematopoiesis-191120.rds")
col_healthy <- colData(scRNA_healthy)
col_all[rownames(col_healthy), "ProjectClassification"] <- col_healthy$BioClassification
colData(scRNA_all) <- col_all
saveHDF5SummarizedExperiment(scRNA_all, "data/scRNA", replace = T)
write.csv(rowData(scRNA_all), "data/scRNA/var.csv", row.names = F)
write.csv(colData(scRNA_all), "data/scRNA/obs.csv")


scATAC_all <- readRDS("data/scATAC-Cicero-GA-Hematopoiesis-MPAL-191120.rds")
col_all <- colData(scATAC_all)
scATAC_healthy <- readRDS("data/scATAC-Cicero-GA-Hematopoiesis-191120.rds")
col_healthy <- colData(scATAC_healthy)
col_all[rownames(col_healthy), "ProjectClassification"] <- col_healthy$BioClassification
colData(scATAC_all) <- col_all
saveHDF5SummarizedExperiment(scATAC_all, "data/scATAC", replace = T)
write.csv(rowData(scATAC_all), "data/scATAC/var.csv", row.names = F)
write.csv(colData(scATAC_all), "data/scATAC/obs.csv")


scATAC_all <- readRDS("data/scATAC-All-Hematopoiesis-MPAL-191120.rds")
col_all <- colData(scATAC_all)
scATAC_healthy <- readRDS("data/scATAC-Healthy-Hematopoiesis-191120.rds")
col_healthy <- colData(scATAC_healthy)
col_all[rownames(col_healthy), "ProjectClassification"] <- col_healthy$BioClassification
colData(scATAC_all) <- col_all
DropletUtils::write10xCounts(x = assay(scATAC_all), path = "data/scATAC")
write.csv(rowData(scATAC_all), "data/scATAC/var.csv", row.names = F)
write.csv(colData(scATAC_all), "data/scATAC/obs.csv")
