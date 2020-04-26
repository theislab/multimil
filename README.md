# Download data
First, Download the [MPAL 2019](https://github.com/GreenleafLab/MPAL-Single-Cell-2019) data. Specificly, we need
these files to be downloaded:

- [scATAC-seq Hematopoeisis cell x peak Summarized Experiment](https://jeffgranja.s3.amazonaws.com/MPAL-10x/Supplementary_Data/Healthy-Data/scATAC-Healthy-Hematopoiesis-191120.rds)
- [scATAC-seq Hematopoeisis cell x gene activity Summarized Experiment](https://jeffgranja.s3.amazonaws.com/MPAL-10x/Supplementary_Data/Healthy-Data/scATAC-Cicero-GA-Hematopoiesis-191120.rds)
- [scRNA-seq Hematopoeisis cell x gene Summarized Experiment](https://jeffgranja.s3.amazonaws.com/MPAL-10x/Supplementary_Data/Healthy-Data/scRNA-Healthy-Hematopoiesis-191120.rds)
- [scATAC-seq Hematopoeisis + MPAL cell x peak Summarized Experiment](https://jeffgranja.s3.amazonaws.com/MPAL-10x/Supplementary_Data/Healthy-Disease-Data/scATAC-All-Hematopoiesis-MPAL-191120.rds)
- [scATAC-seq Hematopoeisis + MPAL cell x gene activity Summarized Experiment](https://jeffgranja.s3.amazonaws.com/MPAL-10x/Supplementary_Data/Healthy-Disease-Data/scATAC-Cicero-GA-Hematopoiesis-MPAL-191120.rds)
- [scRNA-seq Hematopoeisis + MPAL cell x gene Summarized Experiment](https://jeffgranja.s3.amazonaws.com/MPAL-10x/Supplementary_Data/Healthy-Disease-Data/scRNA-All-Hematopoiesis-MPAL-191120.rds)

After downloading them, put them all in `data/` directory.

Then run these scripts to convert RDS files to 10x matrices which we can import in Python:

```bash
Rscript code/scrna_rds_to_10x.R
Rscript code/scatac_peaks_rds_to_10x.R
Rscript code/scatac_geneactivity_rds_to_10x.R
```

Now, you should have directories `data/scRNA/` and `data/scATAC/peaks/` and `data/scATAC/geneactivity/` filled with the corresponding data.

# Preprocess data
Run these notebooks which load and preprocess the data and save the results in `.h5` format:

- `analysis/scRNA_analysis.ipynb`
- `analysis/scATAC_peaks_analysis.ipynb`
- `analysis/scATAC_geneactivity_analysis.ipynb`

