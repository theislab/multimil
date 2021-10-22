import scanpy as sc
import anndata as ad
import pandas as pd

def organize_multiome_anndatas(
    adatas,
    groups,
    layers=None
):
    # set .X to the desired layer
    # TOOD: add checks for layers
    for mod, (modality_adatas, modality_groups) in enumerate(zip(adatas, groups)):
        for i, (adata, group) in enumerate(zip(modality_adatas, modality_groups)):
            if layers:
                if layer := layers[mod][i]:
                    adata.X = adata.layers[layer].A
            adata.obs['group'] = group


    # concat adatas per modality
    mod_adatas = []
    for modality_adatas in adatas:
        mod_adatas.append(modality_adatas[0])

    for mod, modality_adatas in enumerate(adatas):
        for i in range(1, len(modality_adatas)):
            mod_adatas[mod] = mod_adatas[mod].concatenate(modality_adatas[i], batch_key='concat_batch', index_unique=None)

    # concat modality adatas
    multiome_anndata = sc.AnnData(mod_adatas[0], obs=None, var=None)
    batches = mod_adatas[0].obs['group'].astype('int')

    for i in range(1, len(mod_adatas)):
        adata = mod_adatas[i]
        multiome_anndata = ad.concat([multiome_anndata.T, adata.T], join='outer', fill_value=0).T # hack to concat modality adatas along var axis
        batches = pd.concat([batches, adata.obs['group']], axis=1, ignore_index=True)
        batches[0] = batches[0].fillna(batches[i])

    if batches[0].isnull().values.any():
        print("some of the batches are NaN's, sth's wrong!")

    multiome_anndata.obs['group'] = batches[0].astype('category')

    return multiome_anndata
