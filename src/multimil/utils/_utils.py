from math import ceil

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from matplotlib import pyplot as plt


def create_df(pred, columns=None, index=None) -> pd.DataFrame:
    """Create a pandas DataFrame from a list of predictions.

    Parameters
    ----------
    pred
        List of predictions.
    columns
        Column names, i.e. class_names.
    index
        Index names, i.e. obs_names.

    Returns
    -------
    DataFrame with predictions.
    """
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


def setup_ordinal_regression(adata, ordinal_regression_order, categorical_covariate_keys):
    """Setup ordinal regression.

    Parameters
    ----------
    adata
        Annotated data object.
    ordinal_regression_order
        Order of categories for ordinal regression.
    categorical_covariate_keys
        Keys of categorical covariates.
    """
    if ordinal_regression_order is not None:
        if not set(ordinal_regression_order.keys()).issubset(categorical_covariate_keys):
            raise ValueError(
                f"All keys {ordinal_regression_order.keys()} has to be registered as categorical covariates too, but categorical_covariate_keys = {categorical_covariate_keys}"
            )
        for key in ordinal_regression_order.keys():
            # Get unique values from the column without assuming it's categorical
            unique_values = np.unique(adata.obs[key].values)
            if set(unique_values) != set(ordinal_regression_order[key]):
                raise ValueError(
                    f"Unique values in adata.obs[{key}]={unique_values} are not the same as categories specified = {ordinal_regression_order[key]}"
                )
            adata.obs[key] = adata.obs[key].cat.reorder_categories(ordinal_regression_order[key])


def select_covariates(covs, prediction_idx, n_samples_in_batch) -> torch.Tensor:
    """Select prediction covariates from all covariates.

    Parameters
    ----------
    covs
        Covariates.
    prediction_idx
        Index of predictions.
    n_samples_in_batch
        Number of samples in the batch.

    Returns
    -------
    Prediction covariates.
    """
    if len(prediction_idx) > 0:
        covs = torch.index_select(covs, 1, prediction_idx)
        covs = covs.view(n_samples_in_batch, -1, len(prediction_idx))[:, 0, :]
    else:
        covs = torch.tensor([])
    return covs


def prep_minibatch(covs, sample_batch_size) -> tuple[int, int]:
    """Prepare minibatch.

    Parameters
    ----------
    covs
        Covariates.
    sample_batch_size
        Sample batch size.

    Returns
    -------
    Batch size and number of samples in the batch.
    """
    batch_size = covs.shape[0]

    if batch_size % sample_batch_size != 0:
        n_samples_in_batch = 1
    else:
        n_samples_in_batch = batch_size // sample_batch_size
    return batch_size, n_samples_in_batch


def get_predictions(
    prediction_idx, pred_values, true_values, size, bag_pred, bag_true, full_pred, offset=0
) -> tuple[dict, dict, dict]:
    """Get predictions.

    Parameters
    ----------
    prediction_idx
        Index of predictions.
    pred_values
        Predicted values.
    true_values
        True values.
    size
        Size of the bag minibatch.
    bag_pred
        Bag predictions.
    bag_true
        Bag true values.
    full_pred
        Full predictions, i.e. on cell-level.
    offset
        Offset, needed because of several possible types of predictions.

    Returns
    -------
    Bag predictions, bag true values, full predictions on cell-level.
    """
    for i in range(len(prediction_idx)):
        bag_pred[i] = bag_pred.get(i, []) + [pred_values[offset + i].cpu()]
        bag_true[i] = bag_true.get(i, []) + [true_values[:, i].cpu()]
        # TODO in ord reg had pred[len(self.mil.class_idx) + i].repeat(1, size).flatten()
        # in reg had
        # cell level, i.e. prediction for the cell = prediction for the bag
        full_pred[i] = full_pred.get(i, []) + [pred_values[offset + i].unsqueeze(1).repeat(1, size, 1).flatten(0, 1)]
    return bag_pred, bag_true, full_pred


def get_bag_info(bags, n_samples_in_batch, minibatch_size, cell_counter, bag_counter, sample_batch_size):
    """Get bag information.

    Parameters
    ----------
    bags
        Bags.
    n_samples_in_batch
        Number of samples in the batch.
    minibatch_size
        Minibatch size.
    cell_counter
        Cell counter.
    bag_counter
        Bag counter.
    sample_batch_size
        Sample batch size.

    Returns
    -------
    Updated bags, cell counter, and bag counter.
    """
    if n_samples_in_batch == 1:
        bags += [[bag_counter] * minibatch_size]
        cell_counter += minibatch_size
        bag_counter += 1
    else:
        bags += [[bag_counter + i] * sample_batch_size for i in range(n_samples_in_batch)]
        bag_counter += n_samples_in_batch
        cell_counter += sample_batch_size * n_samples_in_batch
    return bags, cell_counter, bag_counter


def save_predictions_in_adata(
    adata, idx, predictions, bag_pred, bag_true, cell_pred, class_names, name, clip, reg=False
):
    """Save predictions in anndata object.

    Parameters
    ----------
    adata
        Annotated data object.
    idx
        Index, i.e. obs_names.
    predictions
        Predictions.
    bag_pred
        Bag predictions.
    bag_true
        Bag true values.
    cell_pred
        Cell predictions.
    class_names
        Class names.
    name
        Name of the prediction column.
    clip
        Whether to transofrm the predictions. One of `clip`, `argmax`, or `none`.
    reg
        Whether the rediciton task is a regression task.
    """
    # cell level predictions)

    if clip == "clip":  # ord regression
        df = create_df(cell_pred[idx], [name], index=adata.obs_names)
        adata.obsm[f"full_predictions_{name}"] = df
        adata.obs[f"predicted_{name}"] = np.clip(np.round(df.to_numpy()), a_min=0.0, a_max=len(class_names) - 1.0)
    elif clip == "argmax":  # classification
        df = create_df(cell_pred[idx], class_names, index=adata.obs_names)
        adata.obsm[f"full_predictions_{name}"] = df
        adata.obs[f"predicted_{name}"] = df.to_numpy().argmax(axis=1)
    else:  # regression
        df = create_df(cell_pred[idx], [name], index=adata.obs_names)
        adata.obsm[f"full_predictions_{name}"] = df
        adata.obs[f"predicted_{name}"] = df.to_numpy()
    if reg is False:
        adata.obs[f"predicted_{name}"] = adata.obs[f"predicted_{name}"].astype("category")
        adata.obs[f"predicted_{name}"] = adata.obs[f"predicted_{name}"].cat.rename_categories(
            dict(enumerate(class_names))
        )

    # bag level predictions
    adata.uns[f"bag_true_{name}"] = create_df(bag_true, predictions)
    if clip == "clip":  # ordinal regression
        df_bag = create_df(bag_pred[idx], [name])
        adata.uns[f"bag_full_predictions_{name}"] = np.clip(
            np.round(df_bag.to_numpy()), a_min=0.0, a_max=len(class_names) - 1.0
        )
    elif clip == "argmax":  # classification
        df_bag = create_df(bag_pred[idx], class_names)
        adata.uns[f"bag_full_predictions_{name}"] = df_bag.to_numpy().argmax(axis=1)
    else:  # regression
        df_bag = create_df(bag_pred[idx], [name])
        adata.uns[f"bag_full_predictions_{name}"] = df_bag.to_numpy()


def plt_plot_losses(history, loss_names, save):
    """Plot losses.

    Parameters
    ----------
    history
        History of losses.
    loss_names
        Loss names to plot.
    save
        Path to save the plot.
    """
    df = pd.concat(history, axis=1)
    df.columns = df.columns.droplevel(-1)
    df["epoch"] = df.index

    nrows = ceil(len(loss_names) / 2)

    plt.figure(figsize=(15, 5 * nrows))

    for i, name in enumerate(loss_names):
        plt.subplot(nrows, 2, i + 1)
        plt.plot(df["epoch"], df[name + "_train"], ".-", label=name + "_train")
        plt.plot(df["epoch"], df[name + "_validation"], ".-", label=name + "_validation")
        plt.xlabel("epoch")
        plt.legend()
    if save is not None:
        plt.savefig(save, bbox_inches="tight")


def get_sample_representations(
    adata,
    sample_key,
    use_rep="X",
    aggregation="weighted",
    cell_attn_key="cell_attn",
    covs_to_keep=None,
    top_fraction=None,
) -> ad.AnnData:
    """Get sample representations from cell-level representations.

    Parameters
    ----------
    adata
        Annotated data object with cell-level representations.
    sample_key
        Key in `adata.obs` that identifies samples.
    use_rep
        Key in `adata.obsm` to use for sample representations or '.X' (default is 'X').
    aggregation
        Method to aggregate cell-level representations to sample-level. Options are 'weighted' or 'mean'.
    cell_attn_key
        Key in `adata.obs` that contains cell-level attention weights (if aggregation is 'weighted').
    covs_to_keep
        List of sample-level covariate keys to keep in the final sample representation.
    top_fraction
        Fraction of top cells to select based on attention weights. If None, uses all cells.
        If provided, will first score top cells and then use only those for sample representation.

    Returns
    -------
    ad.AnnData
        Annotated data object with sample-level representations.
    """
    if use_rep == "X":
        tmp = adata.copy()
    else:
        if use_rep not in adata.obsm.keys():
            raise ValueError(f"Key '{use_rep}' not found in adata.obsm. Available keys: {adata.obsm.keys()}")
        tmp = sc.AnnData(adata.obsm[use_rep], obs=adata.obs.copy())
    tmp.obs[sample_key] = tmp.obs[sample_key].astype(str)

    # If top_fraction is provided, first score top cells and filter
    if top_fraction is not None:
        if cell_attn_key not in tmp.obs.columns:
            raise ValueError(f"Key '{cell_attn_key}' not found in adata.obs. Required for top cell selection.")

        # Use the existing score_top_cells function
        score_top_cells(tmp, top_fraction=top_fraction, sample_key=sample_key, key_added="_top_cell_flag")

        # Filter to only top cells
        tmp = tmp[tmp.obs["_top_cell_flag"]].copy()
        tmp.obs = tmp.obs.drop("_top_cell_flag", axis=1)

    for i in range(tmp.X.shape[1]):
        if aggregation == "weighted":
            tmp.obs[f"latent{i}"] = tmp.X[:, i] * tmp.obs[cell_attn_key]
        elif aggregation == "mean":
            tmp.obs[f"latent{i}"] = tmp.X[:, i].copy()
        else:
            raise ValueError(f"Aggregation method {aggregation} is not supported. Use 'weighted' or 'mean'.")

    if covs_to_keep is not None:
        # check that covariates are sample-level i.e. have the same value for all cells in a sample and print warning if not and which ones are not
        for cov in covs_to_keep:
            if cov not in tmp.obs.columns:
                raise ValueError(f"Covariate '{cov}' not found in adata.obs. Available keys: {tmp.obs.columns}")
            # check that value is the same for all cells in each sample
            if tmp.obs.groupby(sample_key)[cov].nunique().max() > 1:
                raise ValueError(
                    f"Covariate '{cov}' has different values for different cells in a sample. "
                    "Please pass only sample-level covariates."
                )
        if sample_key not in covs_to_keep:
            covs_to_keep = [sample_key] + covs_to_keep
    else:
        covs_to_keep = [sample_key]

    if aggregation == "weighted":
        df = (
            tmp.obs[[f"latent{i}" for i in range(tmp.X.shape[1])] + [sample_key]].groupby(sample_key).agg("sum")
        )  # because already multiplied by normalized weights
    elif aggregation == "mean":
        df = tmp.obs[[f"latent{i}" for i in range(tmp.X.shape[1])] + [sample_key]].groupby(sample_key).agg("mean")
    df = df.join(tmp.obs[covs_to_keep].groupby(sample_key).agg("first"))

    final_covs = [cov for cov in covs_to_keep if cov != sample_key]
    pb = sc.AnnData(df.drop(final_covs, axis=1).values)
    pb.obs = df[final_covs].copy()

    return pb


def score_top_cells(adata, top_fraction=0.1, sample_key=None, key_added="top_cell_attn"):
    """Score top cells based on cell attention weights.

    Parameters
    ----------
    adata
        Annotated data object with cell attention weights in `adata.obs['cell_attn']`.
    top_fraction
        Fraction of top cells to select based on attention weights (default is 0.1).
    sample_key
        Key in `adata.obs` that identifies samples. If None, will calculate across all cells.
    key_added
        Key in `adata.obs` to store the top cell attention scores (default is 'top_cell_attn').
    """
    if "cell_attn" not in adata.obs.columns:
        raise ValueError("adata.obs must contain 'cell_attn' column with cell attention weights.")

    adata.obs[key_added] = False
    if sample_key is None:
        sample_key = "_tmp_sample"
        adata.obs[sample_key] = "_tmp_sample"
    for sample in np.unique(adata.obs[sample_key]):
        adata_sample = adata[adata.obs[sample_key] == sample].copy()
        threshold_idx = int(len(adata_sample) * (1 - top_fraction))
        threshold_value = sorted(adata_sample.obs["cell_attn"])[threshold_idx]
        top_idx = adata_sample[adata_sample.obs["cell_attn"] >= threshold_value].obs_names
        adata.obs.loc[top_idx, key_added] = True
    if sample_key == "_tmp_sample":
        del adata.obs[sample_key]
