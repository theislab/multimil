import pandas as pd
import torch
import scipy
import numpy as np
from matplotlib import pyplot as plt
from math import ceil

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

def select_covariates(covs, prediction_idx, n_samples_in_batch):
    if len(prediction_idx) > 0:
        covs = torch.index_select(covs, 1, prediction_idx)
        covs = covs.view(
            n_samples_in_batch, -1, len(prediction_idx)
        )[:, 0, :]
    else:
        covs = torch.tensor([]) 
    return covs

def prep_minibatch(covs, sample_batch_size):
    batch_size = covs.shape[0]

    if (batch_size % sample_batch_size != 0):
        n_samples_in_batch = 1
    else:
        n_samples_in_batch = batch_size // sample_batch_size 
    return batch_size, n_samples_in_batch

def get_predictions(prediction_idx, pred_values, true_values, size, bag_pred, bag_true, full_pred, offset=0):
    for i in range(len(prediction_idx)):
        bag_pred[i] = bag_pred.get(i, []) + [pred_values[offset + i].cpu()]
        bag_true[i] = bag_true.get(i, []) + [true_values[:, i].cpu()]
        # TODO in ord reg had pred[len(self.mil.class_idx) + i].repeat(1, size).flatten()
        # in reg had 
        # cell level, i.e. prediction for the cell = prediction for the bag
        full_pred[i] = full_pred.get(i, []) + [pred_values[offset + i].unsqueeze(1).repeat(1, size, 1).flatten(0, 1)]
    return bag_pred, bag_true, full_pred

def get_bag_info(bags, n_samples_in_batch, minibatch_size, cell_counter, bag_counter, sample_batch_size):
    if n_samples_in_batch == 1:
        bags += [[bag_counter] * minibatch_size]
        cell_counter += minibatch_size
        bag_counter += 1
    else:
        bags += [[bag_counter + i]*sample_batch_size for i in range(n_samples_in_batch)]
        bag_counter += n_samples_in_batch
        cell_counter += sample_batch_size * n_samples_in_batch
    return bags, cell_counter, bag_counter

def save_predictions_in_adata(adata, idx, predictions, bag_pred, bag_true, cell_pred, class_names, name, clip, reg=False):
        # cell level predictions
        df = create_df(cell_pred[idx], class_names, index=adata.obs_names)
        adata.obsm[f"full_predictions_{name}"] = df
        if clip == 'clip': # ord regression
            adata.obs[f"predicted_{name}"] = np.clip(np.round(df.to_numpy()), a_min=0.0, a_max=len(class_names) - 1.0)
        elif clip == 'argmax': # classification
            adata.obs[f"predicted_{name}"] = df.to_numpy().argmax(axis=1)
        else:
            adata.obs[f"predicted_{name}"] = df.to_numpy()
        if reg is False:
            adata.obs[f"predicted_{name}"] = adata.obs[f"predicted_{name}"].astype(
                "category"
            )
            adata.obs[f"predicted_{name}"] = adata.obs[
                f"predicted_{name}"
            ].cat.rename_categories({i: cl for i, cl in enumerate(class_names)})

        # bag level predictions
        adata.uns[f"bag_true_{name}"] = create_df(
            bag_true, predictions
        )
        df_bag = create_df(bag_pred[idx], class_names)
        if clip == 'clip':
            adata.uns[f"bag_full_predictions_{name}"] = np.clip(np.round(df_bag.to_numpy()), a_min=0.0, a_max=len(class_names) - 1.0)
        elif clip == 'argmax':
            adata.uns[f"bag_full_predictions_{name}"] = df_bag.to_numpy().argmax(axis=1)
        else:
            adata.uns[f"bag_full_predictions_{name}"] = df_bag.to_numpy()

def plt_plot_losses(df, loss_names, save):
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
