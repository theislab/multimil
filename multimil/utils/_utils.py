import pandas as pd
import torch
import scipy

def create_df(pred, columns=None, index=None):
    if isinstance(pred, dict):
        for key in pred.keys():
            pred[key] = torch.cat(pred[key]).squeeze().cpu().numpy()
    else:
        pred = torch.cat(pred).squeeze().cpu().numpy()

    df = pd.DataFrame(pred)
    if index is not None:
        df.index = index
    if columns is not None:
        df.columns = columns
    return df

def calculate_size_factor(adata, size_factor_key, rna_indices_end):
    # TODO check that organize_multiome_anndatas was run, i.e. that .uns['modality_lengths'] was added, needed for q2r
    if size_factor_key is not None and rna_indices_end is not None:
        raise ValueError(
            "Only one of [`size_factor_key`, `rna_indices_end`] can be specified, but both are not `None`."
        )
    # TODO change to when both are None and data in unimodal, use all input features to calculate the size factors, add warning 
    if size_factor_key is None and rna_indices_end is None:
        raise ValueError("One of [`size_factor_key`, `rna_indices_end`] has to be specified, but both are `None`.")

    if size_factor_key is not None:
        return size_factor_key
    if rna_indices_end is not None:
        if scipy.sparse.issparse(adata.X):
            adata.obs.loc[:, "size_factors"] = adata[:, :rna_indices_end].X.A.sum(1).T.tolist()
        else:
            adata.obs.loc[:, "size_factors"] = adata[:, :rna_indices_end].X.sum(1).T.tolist()
        return "size_factors"

def setup_ordinal_regression(adata, ordinal_regression_order, categorical_covariate_keys):
    # TODO make sure not to assume categorical columns for ordinal regression -> change to np.unique if needed
    if ordinal_regression_order is not None:
        if not set(ordinal_regression_order.keys()).issubset(
            categorical_covariate_keys
        ):
            raise ValueError(
                f"All keys {ordinal_regression_order.keys()} has to be registered as categorical covariates too, but categorical_covariate_keys = {categorical_covariate_keys}"
            )
        for key in ordinal_regression_order.keys():
            adata.obs[key] = adata.obs[key].astype("category")
            if set(adata.obs[key].cat.categories) != set(
                ordinal_regression_order[key]
            ):
                raise ValueError(
                    f"Categories of adata.obs[{key}]={adata.obs[key].cat.categories} are not the same as categories specified = {ordinal_regression_order[key]}"
                )
            adata.obs[key] = adata.obs[key].cat.reorder_categories(
                ordinal_regression_order[key]
            )



