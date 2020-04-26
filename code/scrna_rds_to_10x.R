library(SummarizedExperiment)
library(DropletUtils)

print('starting processing scRNA data')
scRNA_all <- readRDS("data/scRNA-All-Hematopoiesis-MPAL-191120.rds")
col_all <- colData(scRNA_all)
scRNA_healthy <- readRDS("data/scRNA-Healthy-Hematopoiesis-191120.rds")
col_healthy <- colData(scRNA_healthy)
col_all[rownames(col_healthy), "ProjectClassification"] <- col_healthy$BioClassification
colData(scRNA_all) <- col_all
DropletUtils::write10xCounts(x = assay(scRNA_all), path = "data/scRNA")
write.csv(rowData(scRNA_all), "data/scRNA/var.csv", row.names = F)
write.csv(colData(scRNA_all), "data/scRNA/obs.csv")
print('saved scRNA-seq data in data/scRNA')
