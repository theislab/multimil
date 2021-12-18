import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np

def organize_multiome_anndatas(
    adatas,
    groups,
    layers=None,
    modality_lengths=None
):
    # set .X to the desired layer
    # TOOD: add checks for layers

    # needed for scArches operation setup
    datasets_lengths = {}
    datasets_groups = {}
    datasets_obs_names = {}
    datasets_obs = {}
    for mod, (modality_adatas, modality_groups) in enumerate(zip(adatas, groups)):
        for i, (adata, group) in enumerate(zip(modality_adatas, modality_groups)):
            if adata:
                datasets_lengths[i] = len(adata)
                datasets_groups[i] = group
                datasets_obs_names[i] = adata.obs_names
                datasets_obs[i] = adata.obs

    # TODO: add check that obs_names are same for the same groups

    for mod, (modality_adatas, modality_groups) in enumerate(zip(adatas, groups)):
        for i, (adata, group) in enumerate(zip(modality_adatas, modality_groups)):
            if not isinstance(adata, ad.AnnData) and adata == None:
                adatas[mod][i] = ad.AnnData(np.zeros((datasets_lengths[i], modality_lengths[mod])))
                adatas[mod][i].obs_names = datasets_obs_names[i]
                adatas[mod][i].obs = datasets_obs[i]
                groups[mod][i] = datasets_groups[i]
            if layers:
                if layers[mod][i]:
                    layer = layers[mod][i]
                    adatas[mod][i].X = adatas[mod][i].layers[layer].A
            adatas[mod][i].obs['group'] = datasets_groups[i]

    # concat adatas per modality
    mod_adatas = []
    # first in list
    for modality_adatas in adatas:
        mod_adatas.append(modality_adatas[0])
    # the rest
    for mod, modality_adatas in enumerate(adatas):
        for i in range(1, len(modality_adatas)):
            mod_adatas[mod] = mod_adatas[mod].concatenate(modality_adatas[i], batch_key='concat_batch', index_unique=None)

    # concat modality adatas
    multiome_anndata = mod_adatas[0]
    batches = mod_adatas[0].obs['group'].astype('int')

    for i in range(1, len(mod_adatas)):
        adata = mod_adatas[i]
        multiome_anndata = ad.concat([multiome_anndata.T, adata.T], join='outer', fill_value=0).T # hack to concat modality adatas along var axis

        batches = pd.concat([batches, adata.obs['group']], axis=1, ignore_index=True)
        batches[0] = batches[0].fillna(batches[i])


    if batches[0].isnull().values.any():
        print("some of the batches are NaN's, sth's wrong!")

    multiome_anndata.obs = mod_adatas[0].obs
    multiome_anndata.obs['group'] = batches[0].astype('category')

    return multiome_anndata
