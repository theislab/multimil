library(SummarizedExperiment)
library(DropletUtils)

print('starting processing scATAC gene activity data')
scATAC_all <- readRDS("data/scATAC-Cicero-GA-Hematopoiesis-MPAL-191120.rds")
col_all <- colData(scATAC_all)
scATAC_healthy <- readRDS("data/scATAC-Cicero-GA-Hematopoiesis-191120.rds")
col_healthy <- colData(scATAC_healthy)
col_all[rownames(col_healthy), "ProjectClassification"] <- col_healthy$BioClassification
colData(scATAC_all) <- col_all
DropletUtils::write10xCounts(x = assay(scATAC_all), path = "data/scATAC/geneactivity")
write.csv(rowData(scATAC_all), "data/scATAC/geneactivity/var.csv")
write.csv(colData(scATAC_all), "data/scATAC/geneactivity/obs.csv")
print('saved scATAC-seq gene activity data in data/scATAC/geneactivity')
