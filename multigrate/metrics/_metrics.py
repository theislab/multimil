# adjusted from scIB
# on 11 November 2021
# https://github.com/theislab/scib/blob/985d8155391fdfbddec024de428308b5a57ee280/scib/metrics/metrics.py

import numpy as np
import pandas as pd

from ..utils._utils import check_adata, check_batch
from ._ari import ari
from ._clustering import opt_louvain
from ._graph_connectivity import graph_connectivity
from ._isolated_labels import isolated_labels
from ._nmi import nmi
from ._silhouette import silhouette, silhouette_batch


def metrics(
        adata,
        batch_key,
        label_key,
        **kwargs
):
    """
    Only MultiMIL metrics:

    Biological conservation
        Cell type ASW
        Isolated label ASW
        NMI cluster/label
        ARI cluster/label

    Batch conservation
        Graph connectivity
        Batch ASW
    """
    return metrics_icb(
        adata,
        batch_key,
        label_key,
        isolated_labels_asw_=True,
        nmi_=True,
        ari_=True,
        silhouette_=True,
        graph_conn_=True,
        **kwargs
    )

def metrics_icb(
        adata_int,
        batch_key,
        label_key,
        cluster_key='cluster',
        cluster_nmi=None,
        ari_=False,
        nmi_=False,
        nmi_method='arithmetic',
        nmi_dir=None,
        silhouette_=False,
        embed='X_pca',
        si_metric='euclidean',
        isolated_labels_asw_=False,
        n_isolated=None,
        graph_conn_=False,
        verbose=False,
):

    check_adata(adata_int)
    check_batch(batch_key, adata_int.obs)
    check_batch(label_key, adata_int.obs)

    # clustering
    if nmi_ or ari_:
        res_max, nmi_max, nmi_all = opt_louvain(
            adata_int,
            label_key=label_key,
            cluster_key=cluster_key,
            function=nmi,
            plot=False,
            verbose=False,
            use_rep=embed,
            inplace=True,
            force=True
        )
        if cluster_nmi is not None:
            nmi_all.to_csv(cluster_nmi, header=False)
            print(f'saved clustering NMI values to {cluster_nmi}')

    results = {}

    if nmi_:
        print('NMI...')
        nmi_score = nmi(
            adata_int,
            group1=cluster_key,
            group2=label_key,
            method=nmi_method,
            nmi_dir=nmi_dir
        )
    else:
        nmi_score = np.nan

    if ari_:
        print('ARI...')
        ari_score = ari(
            adata_int,
            group1=cluster_key,
            group2=label_key
        )
    else:
        ari_score = np.nan

    if silhouette_:
        print('Silhouette score...')
        # global silhouette coefficient
        asw_label = silhouette(
            adata_int,
            group_key=label_key,
            embed=embed,
            metric=si_metric
        )
        # silhouette coefficient per batch
        asw_batch = silhouette_batch(
            adata_int,
            batch_key=batch_key,
            group_key=label_key,
            embed=embed,
            metric=si_metric,
            return_all=False,
            verbose=False
        )
    else:
        asw_label = np.nan
        asw_batch = np.nan

    if isolated_labels_asw_:
        print("Isolated labels ASW...")
        il_score_asw = isolated_labels(
            adata_int,
            label_key=label_key,
            batch_key=batch_key,
            embed=embed,
            cluster=False,
            iso_threshold=n_isolated,
            verbose=False
        ) if silhouette_ else np.nan
    else:
        il_score_asw = np.nan

    if graph_conn_:
        print('Graph connectivity...')
        graph_conn_score = graph_connectivity(
            adata_int,
            label_key=label_key
        )
    else:
        graph_conn_score = np.nan

    results = {
        'NMI_cluster/label': nmi_score,
        'ARI_cluster/label': ari_score,
        'ASW_label': asw_label,
        'ASW_label/batch': asw_batch,
        'isolated_label_silhouette': il_score_asw,
        'graph_conn': graph_conn_score,
    }

    return pd.DataFrame.from_dict(results, orient='index')
