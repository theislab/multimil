import warnings
from typing import List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd


def organize_multiome_anndatas(
    adatas: List[List[Union[ad.AnnData, None]]],
    layers: Optional[List[List[Union[str, None]]]] = None,
):
    """Concatenate all the input anndata objects.

    These anndata objects should already have been preprocessed so that all single-modality
    objects use a subset of the features used in the multiome object. The feature names (index of
    `.var`) should match between the objects for vertical integration and cell names (index of
    `.obs`) should match between the objects for horizontal integration.

    :param adatas:
        List of Lists with AnnData objects or None where each sublist corresponds to a modality
    :param layers:
        List of Lists of the same lengths as `adatas` specifying which `.layer` to use for each AnnData. Default is None which means using `.X`.

    """
    # TOOD: add checks for layers

    # needed for scArches operation setup
    datasets_lengths = {}
    datasets_obs_names = {}
    datasets_obs = {}
    modality_lengths = {}
    modality_var_names = {}

    # sanity checks and preparing data for concat
    for mod, modality_adatas in enumerate(adatas):
        for i, adata in enumerate(modality_adatas):
            if adata is not None:
                # will create .obs['group'] later, so throw a warning here if the column already exists
                if "group" in adata.obs.columns:
                    warnings.warn(
                        "Column `.obs['group']` will be overwritten. Please save the original data in another column if needed.",
                        stacklevel=2,
                    )
                # check that all adatas in the same modality have the same number of features
                if (mod_length := modality_lengths.get(mod, None)) is None:
                    modality_lengths[mod] = adata.shape[1]
                else:
                    if adata.shape[1] != mod_length:
                        raise ValueError(
                            f"Adatas have different number of features for modality {mod}, namely {mod_length} and {adata.shape[1]}."
                        )
                # check that there is the same number of observations for paired data
                if (dataset_length := datasets_lengths.get(i, None)) is None:
                    datasets_lengths[i] = adata.shape[0]
                else:
                    if adata.shape[0] != dataset_length:
                        raise ValueError(
                            f"Paired adatas have different number of observations for group {i}, namely {dataset_length} and {adata.shape[0]}."
                        )
                # check that .obs_names are the same for paired data
                if (dataset_obs_names := datasets_obs_names.get(i, None)) is None:
                    datasets_obs_names[i] = adata.obs_names
                else:
                    if np.sum(adata.obs_names != dataset_obs_names):
                        raise ValueError(f"`.obs_names` are not the same for group {i}.")
                # keep all the .obs
                if datasets_obs.get(i, None) is None:
                    datasets_obs[i] = adata.obs
                    datasets_obs[i].loc[:, "group"] = i
                else:
                    cols_to_use = adata.obs.columns.difference(datasets_obs[i].columns)
                    datasets_obs[i] = datasets_obs[i].join(adata.obs[cols_to_use])
                modality_var_names[mod] = adata.var_names

    for mod, modality_adatas in enumerate(adatas):
        for i, adata in enumerate(modality_adatas):
            if not isinstance(adata, ad.AnnData) and adata is None:
                X_zeros = np.zeros((datasets_lengths[i], modality_lengths[mod]))
                adatas[mod][i] = ad.AnnData(X_zeros, dtype=X_zeros.dtype)
                adatas[mod][i].obs_names = datasets_obs_names[i]
                adatas[mod][i].var_names = modality_var_names[mod]
                adatas[mod][i] = adatas[mod][i].copy()
            if layers is not None:
                if layers[mod][i]:
                    layer = layers[mod][i]
                    adatas[mod][i] = adatas[mod][i].copy()
                    adatas[mod][i].X = adatas[mod][i].layers[layer].copy()

    # concat adatas within each modality first
    mod_adatas = []
    for modality_adatas in adatas:
        mod_adatas.append(ad.concat(modality_adatas, join="outer"))

    # concat modality adatas along the feature axis
    multiome_anndata = ad.concat(mod_adatas, axis=1, label="modality")

    # add .obs back
    multiome_anndata.obs = pd.concat(datasets_obs.values())

    # we will need modality_length later for the model init
    multiome_anndata.uns["modality_lengths"] = modality_lengths
    multiome_anndata.var_names_make_unique()

    return multiome_anndata
