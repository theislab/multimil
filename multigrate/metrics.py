import numpy as np
import pandas as pd
import scanpy as sc
import sklearn
from scipy.stats import itemfreq, entropy
from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_samples, pair_confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .utils import remove_sparsity

def __entropy_from_indices(indices):
    return entropy(np.array(itemfreq(indices)[:, 1].astype(np.int32)))


def entropy_batch_mixing(adata, label_key='modal',
                         n_neighbors=50, n_pools=50, n_samples_per_pool=100):
    """Computes Entory of Batch mixing metric for ``adata`` given the batch column name.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
        label_key: str
            Name of the column which contains information about different studies in ``adata.obs`` data frame.
        n_neighbors: int
            Number of nearest neighbors.
        n_pools: int
            Number of EBM computation which will be averaged.
        n_samples_per_pool: int
            Number of samples to be used in each pool of execution.

        Returns
        -------
        score: float
            EBM score. A float between zero and one.

    """
    adata = remove_sparsity(adata)

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.X)
    indices = neighbors.kneighbors(adata.X, return_distance=False)[:, 1:]
    batch_indices = np.vectorize(lambda i: adata.obs[label_key].values[i])(indices)

    entropies = np.apply_along_axis(__entropy_from_indices, axis=1, arr=batch_indices)

    # average n_pools entropy results where each result is an average of n_samples_per_pool random samples.
    if n_pools == 1:
        score = np.mean(entropies)
    else:
        score = np.mean([
            np.mean(entropies[np.random.choice(len(entropies), size=n_samples_per_pool)])
            for _ in range(n_pools)
        ])

    return score


def asw(adata, label_key='modal'):
    """Computes Average Silhouette Width (ASW) metric for ``adata`` given the batch column name.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
        label_key: str
            Name of the column which contains information about different studies in ``adata.obs`` data frame.
        Returns
        -------
        score: float
            ASW score. A float between -1 and 1.

    """
    adata = remove_sparsity(adata)
    labels = adata.obs[label_key].values
    labels_encoded = LabelEncoder().fit_transform(labels)
    return silhouette_score(adata.X, labels_encoded).item()

def silhouette(adata, group_key, metric='euclidean', embed='X_pca', scale=True):
    """
    wrapper for sklearn silhouette function values range from [-1, 1] with 1 being an ideal fit, 0 indicating overlapping clusters and -1 indicating misclassified cells
    """
    if embed not in adata.obsm.keys():
        print(adata.obsm.keys())
        raise KeyError(f'{embed} not in obsm')
    asw = sklearn.metrics.silhouette_score(adata.obsm[embed], adata.obs[group_key], metric=metric)
    if scale:
        asw = (asw + 1)/2
    return asw

def silhouette_batch(adata, batch_key, group_key, metric='euclidean',
                     embed='X_pca', verbose=False, scale=True):
    """
    Silhouette score of batch labels subsetted for each group.
    params:
        batch_key: batches to be compared against
        group_key: group labels to be subsetted by e.g. cell type
        metric: see sklearn silhouette score
        embed: name of column in adata.obsm
    returns:
        all scores: absolute silhouette scores per group label
        group means: if `mean=True`
    """
    if embed not in adata.obsm.keys():
        if embed == 'X_pca':
            sc.tl.pca(adata)
        else:
            print(adata.obsm.keys())
            raise KeyError(f'{embed} not in obsm')

    sil_all = pd.DataFrame(columns=['group', 'silhouette_score'])
    for group in adata.obs[group_key].unique():
        adata_group = adata[adata.obs[group_key] == group]
        if adata_group.obs[batch_key].nunique() == 1:
            continue
        if len(adata_group.obs_names) <= 2:
            continue
        sil_per_group = sklearn.metrics.silhouette_samples(adata_group.obsm[embed], adata_group.obs[batch_key],
                                                           metric=metric)
        # take only absolute value
        sil_per_group = [abs(i) for i in sil_per_group]
        if scale:
            # scale s.t. highest number is optimal
            sil_per_group = [1 - i for i in sil_per_group]
        d = pd.DataFrame({'group' : [group]*len(sil_per_group), 'silhouette_score' : sil_per_group})
        sil_all = sil_all.append(d)
    sil_all = sil_all.reset_index(drop=True)
    sil_means = sil_all.groupby('group').mean()

    if verbose:
        print(f'mean silhouette per cell: {sil_means}')
    return sil_all, sil_means

def nmi_old(adata, label_key='modal'):
    adata = remove_sparsity(adata)

    n_labels = len(adata.obs[label_key].unique().tolist())
    kmeans = KMeans(n_labels, n_init=200)

    labels_pred = kmeans.fit_predict(adata.X)
    labels = adata.obs[label_key].values
    labels_encoded = LabelEncoder().fit_transform(labels)

    return normalized_mutual_info_score(labels_encoded, labels_pred)

def nmi(adata, group1, group2, method="arithmetic"):
    """
    Normalized mutual information NMI based on 2 different cluster assignments `group1` and `group2`
    params:
        adata: Anndata object
        group1: column name of `adata.obs` or group assignment
        group2: column name of `adata.obs` or group assignment
        method: NMI implementation
            'max': scikit method with `average_method='max'`
            'min': scikit method with `average_method='min'`
            'geometric': scikit method with `average_method='geometric'`
            'arithmetic': scikit method with `average_method='arithmetic'`
        return:
        normalized mutual information (NMI)
    """

    if isinstance(group1, str):
        group1 = adata.obs[group1].tolist()
    elif isinstance(group1, pd.Series):
        group1 = group1.tolist()

    if isinstance(group2, str):
        group2 = adata.obs[group2].tolist()
    elif isinstance(group2, pd.Series):
        group2 = group2.tolist()

    if len(group1) != len(group2):
        raise ValueError(f'different lengths in group1 ({len(group1)}) and group2 ({len(group2)})')

    if method in ['max', 'min', 'geometric', 'arithmetic']:
        nmi_value = sklearn.metrics.normalized_mutual_info_score(group1, group2, average_method=method)
    else:
        raise ValueError(f"Method {method} not valid")
    return nmi_value


def knn_purity(adata, label_key='modal', n_neighbors=30):
    """Computes KNN Purity metric for ``adata`` given the batch column name.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
        label_key: str
            Name of the column which contains information about different studies in ``adata.obs`` data frame.
        n_neighbors: int
            Number of nearest neighbors.
        Returns
        -------
        score: float
            KNN purity score. A float between 0 and 1.

    """
    adata = remove_sparsity(adata)
    labels = LabelEncoder().fit_transform(adata.obs[label_key].to_numpy())

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.X)
    indices = nbrs.kneighbors(adata.X, return_distance=False)[:, 1:]
    neighbors_labels = np.vectorize(lambda i: labels[i])(indices)

    # pre cell purity scores
    scores = ((neighbors_labels - labels.reshape(-1, 1)) == 0).mean(axis=1)
    res = [
        np.mean(scores[labels == i]) for i in np.unique(labels)
    ]  # per cell-type purity

    return np.mean(res)


### ARI adjusted rand index
def ari(adata, group1, group2='louvain'):
    """
    params:
        adata: anndata object
        group1: ground-truth cluster assignments (e.g. cell type labels)
        group2: "predicted" cluster assignments
    The function is symmetric, so group1 and group2 can be switched
    """
    if group2 not in adata.obs:
        sc.tl.louvain(adata)
    group1 = adata.obs[group1].tolist()
    group2 = adata.obs[group2].tolist()

    (tn, fp), (fn, tp) = pair_confusion_matrix(group1, group2)

    # fix overflow
    tn = float(tn)
    fp = float(fp)
    fn = float(fn)
    tp = float(tp)

    if fn == 0 and fp == 0:
        return 1.0

    return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) +
                                       (tp + fp) * (fp + tn))

### Isolated label score
def isolated_labels(adata, label_key, batch_key, cluster_key="iso_cluster",
                    cluster=True, n=None, all_=False, verbose=False, **kwargs):
    """
    score how well labels of isolated labels are distiguished in the dataset by
        1. clustering-based approach
        2. silhouette score
    params:
        cluster: if True, use clustering approach, otherwise use silhouette score approach
        n: max number of batches per label for label to be considered as isolated.
            if n is integer, consider labels that are present for n batches as isolated
            if n=None, consider minimum number of batches that labels are present in
        all_: return scores for all isolated labels instead of aggregated mean
    return:
        by default, mean of scores for each isolated label
        retrieve dictionary of scores for each label if `all_` is specified
    """

    scores = {}
    isolated_labels = get_isolated_labels(adata, label_key, batch_key, cluster_key, n=n, verbose=verbose)
    for label in isolated_labels:
        score = score_isolated_label(adata, label_key, cluster_key, label, cluster=cluster, verbose=verbose, **kwargs)
        scores[label] = score

    if all_:
        return scores
    return np.mean(list(scores.values()))


def get_isolated_labels(adata, label_key, batch_key, cluster_key, n, verbose):
    """
    get labels that are considered isolated by the number of batches
    """

    tmp = adata.obs[[label_key, batch_key]].drop_duplicates()
    batch_per_lab = tmp.groupby(label_key).agg({batch_key: "count"})

    # threshold for determining when label is considered isolated
    if n is None:
        n = batch_per_lab.min().tolist()[0]

    if verbose:
        print(f"isolated labels: no more than {n} batches per label")

    labels = batch_per_lab[batch_per_lab[batch_key] <= n].index.tolist()
    if len(labels) == 0 and verbose:
        print(f"no isolated labels with less than {n} batches")
    return labels

def max_f1(adata, label_key, cluster_key, label, argmax=False):
    """cluster optimizing over largest F1 score of isolated label"""
    obs = adata.obs
    max_cluster = None
    max_f1 = 0
    for cluster in obs[cluster_key].unique():
        y_pred = obs[cluster_key] == cluster
        y_true = obs[label_key] == label
        f1 = sklearn.metrics.f1_score(y_pred, y_true)
        if f1 > max_f1:
            max_f1 = f1
            max_cluster = cluster
    if argmax:
        return max_cluster
    return max_f1


def score_isolated_label(adata, label_key, cluster_key, label, cluster=True, verbose=False, **kwargs):
    """
    compute label score for a single label
    params:
        cluster: if True, use clustering approach, otherwise use silhouette score approach
    """
    adata_tmp = adata.copy()

    def max_label_per_batch(adata, label_key, cluster_key, label, argmax=False):
        """cluster optimizing over cluster with largest number of isolated label per batch"""
        sub = adata.obs[adata.obs[label_key] == label].copy()
        label_counts = sub[cluster_key].value_counts()
        if argmax:
            return label_counts.index[label_counts.argmax()]
        return label_counts.max()

    if cluster:
        opt_louvain(adata_tmp, label_key, cluster_key, function=max_f1, label=label, verbose=False, inplace=True)
        score = max_f1(adata_tmp, label_key, cluster_key, label, argmax=False)
    else:
        adata_tmp.obs['group'] = adata_tmp.obs[label_key] == label
        score = silhouette(adata_tmp, group_key='group', **kwargs)

    del adata_tmp

    if verbose:
        print(f"{label}: {score}")

    return score

def opt_louvain(adata, label_key, cluster_key, function=None, resolutions=None,
                inplace=True, plot=False, force=True, verbose=True, **kwargs):
    """
    params:
        label_key: name of column in adata.obs containing biological labels to be
            optimised against
        cluster_key: name of column to be added to adata.obs during clustering.
            Will be overwritten if exists and `force=True`
        function: function that computes the cost to be optimised over. Must take as
            arguments (adata, group1, group2, **kwargs) and returns a number for maximising
        resolutions: list if resolutions to be optimised over. If `resolutions=None`,
            default resolutions of 20 values ranging between 0.1 and 2 will be used
    returns:
        res_max: resolution of maximum score
        score_max: maximum score
        score_all: `pd.DataFrame` containing all scores at resolutions. Can be used to plot the score profile.
        clustering: only if `inplace=False`, return cluster assignment as `pd.Series`
        plot: if `plot=True` plot the score profile over resolution
    """

    if function is None:
        function = nmi

    if cluster_key in adata.obs.columns:
        if force:
            if verbose:
                print(f"Warning: cluster key {cluster_key} already exists " +
                      "in adata.obs and will be overwritten")
        else:
            raise ValueError(f"cluster key {cluster_key} already exists in " +
                             "adata, please remove the key or choose a different name." +
                             "If you want to force overwriting the key, specify `force=True`")

    if resolutions is None:
        n = 20
        resolutions = [2*x/n for x in range(1,n+1)]

    score_max = 0
    res_max = resolutions[0]
    clustering = None
    score_all = []

    #maren's edit - recompute neighbors if not existing
    try:
        adata.uns['neighbors']
    except KeyError:
        if verbose:
            print('computing neigbours for opt_cluster')
        sc.pp.neighbors(adata)

    for res in resolutions:
        sc.tl.louvain(adata, resolution=res, key_added=cluster_key)
        score = function(adata, label_key, cluster_key, **kwargs)
        score_all.append(score)
        if score_max < score:
            score_max = score
            res_max = res
            clustering = adata.obs[cluster_key]
        del adata.obs[cluster_key]

    if verbose:
        print(f'optimised clustering against {label_key}')
        print(f'optimal cluster resolution: {res_max}')
        print(f'optimal score: {score_max}')

    score_all = pd.DataFrame(zip(resolutions, score_all), columns=('resolution', 'score'))

    if inplace:
        adata.obs[cluster_key] = clustering
        return res_max, score_max, score_all
    else:
        return res_max, score_max, score_all, clustering

### PC Regression
def pcr_comparison(adata_pre, adata_post, covariate, embed=None, n_comps=50, scale=True, verbose=False):
    """
    Compare the effect before and after integration
    Return either the difference of variance contribution before and after integration
    or a score between 0 and 1 (`scaled=True`) with 0 if the variance contribution hasn't
    changed. The larger the score, the more different the variance contributions are before
    and after integration.
    params:
        adata_pre: uncorrected adata
        adata_post: integrated adata
        embed   : if `embed=None`, use the full expression matrix (`adata.X`), otherwise
                  use the embedding provided in `adata_post.obsm[embed]`
        scale: if True, return scaled score
    return:
        difference of R2Var value of PCR
    """

    if embed == 'X_pca':
        embed = None

    pcr_before = pcr(adata_pre, covariate=covariate, recompute_pca=True,
                     n_comps=n_comps, verbose=verbose)
    pcr_after = pcr(adata_post, covariate=covariate, embed=embed, recompute_pca=True,
                    n_comps=n_comps, verbose=verbose)

    if scale:
        score = (pcr_before - pcr_after)/pcr_before
        if score < 0:
            print("Variance contribution increased after integration!")
            print("Setting PCR comparison score to 0.")
            score = 0
        return score
    else:
        return pcr_after - pcr_before

def pcr(adata, covariate, embed=None, n_comps=50, recompute_pca=True, verbose=False):
    """
    PCR for Adata object
    Checks whether to
        + compute PCA on embedding or expression data (set `embed` to name of embedding matrix e.g. `embed='X_emb'`)
        + use existing PCA (only if PCA entry exists)
        + recompute PCA on expression matrix (default)
    params:
        adata: Anndata object
        embed   : if `embed=None`, use the full expression matrix (`adata.X`), otherwise
                  use the embedding provided in `adata_post.obsm[embed]`
        n_comps: number of PCs if PCA should be computed
        covariate: key for adata.obs column to regress against
    return:
        R2Var of PCR
    """
    if verbose:
        print(f"covariate: {covariate}")
    batch = adata.obs[covariate]

    # use embedding for PCA
    if (embed is not None) and (embed in adata.obsm):
        if verbose:
            print(f"compute PCR on embedding n_comps: {n_comps}")
        return pc_regression(adata.obsm[embed], batch, n_comps=n_comps)

    # use existing PCA computation
    elif (recompute_pca == False) and ('X_pca' in adata.obsm) and ('pca' in adata.uns):
        if verbose:
            print("using existing PCA")
        return pc_regression(adata.obsm['X_pca'], batch, pca_var=adata.uns['pca']['variance'])

    # recompute PCA
    else:
        if verbose:
            print(f"compute PCA n_comps: {n_comps}")
        return pc_regression(adata.X, batch, n_comps=n_comps)

def pc_regression(data, variable, pca_var=None, n_comps=50, svd_solver='arpack', verbose=False):
    """
    params:
        data: expression or PCA matrix. Will be assumed to be PCA values, if pca_sd is given
        variable: series or list of batch assignments
        n_comps: number of PCA components for computing PCA, only when pca_sd is not given. If no pca_sd is given and n_comps=None, comute PCA and don't reduce data
        pca_var: iterable of variances for `n_comps` components. If `pca_sd` is not `None`, it is assumed that the matrix contains PCA values, else PCA is computed
    PCA is only computed, if variance contribution is not given (pca_sd).
    """

    if isinstance(data, (np.ndarray, sparse.csr_matrix)):
        matrix = data
    else:
        raise TypeError(f'invalid type: {data.__class__} is not a numpy array or sparse matrix')

    # perform PCA if no variance contributions are given
    if pca_var is None:

        if n_comps is None or n_comps > min(matrix.shape):
            n_comps = min(matrix.shape)

        if n_comps == min(matrix.shape):
            svd_solver = 'full'

        if verbose:
            print("compute PCA")
        pca = sc.tl.pca(matrix, n_comps=n_comps, use_highly_variable=False,
                        return_info=True, svd_solver=svd_solver, copy=True)
        X_pca = pca[0].copy()
        pca_var = pca[3].copy()
        del pca
    else:
        X_pca = matrix
        n_comps = matrix.shape[1]

    ## PC Regression
    if verbose:
        print("fit regression on PCs")

    # handle categorical values
    if pd.api.types.is_numeric_dtype(variable):
        variable = np.array(variable).reshape(-1, 1)
    else:
        if verbose:
            print("one-hot encode categorical values")
        variable = pd.get_dummies(variable)

    # fit linear model for n_comps PCs
    r2 = []
    for i in range(n_comps):
        pc = X_pca[:, [i]]
        lm = sklearn.linear_model.LinearRegression()
        lm.fit(variable, pc)
        r2_score = np.maximum(0,lm.score(variable, pc))
        r2.append(r2_score)

    Var = pca_var / sum(pca_var) * 100
    R2Var = sum(r2*Var)/100

    return R2Var

def graph_connectivity(adata_post, label_key):
    """"
    Metric that quantifies how connected the subgraph corresponding to each batch cluster is.
    """
    if 'neighbors' not in adata_post.uns:
        raise KeyError('Please compute the neighborhood graph before running this '
                       'function!')

    clust_res = []

    for ct in adata_post.obs[label_key].cat.categories:
        adata_post_sub = adata_post[adata_post.obs[label_key].isin([ct]),]
        _,labs = connected_components(adata_post_sub.uns['neighbors']['connectivities'], connection='strong')
        tab = pd.value_counts(labs)
        clust_res.append(tab[0]/sum(tab))

    return np.mean(clust_res)

def metrics(adata_old, adata, batch_key, label_key, asw_label=True, asw_batch=True,
            pcr_batch=True, graph_connectivity_batch=True, nmi_=True, ari_=True,
            isolated_label_asw=True, isolated_label_f1=True, embed='X_pca', save=None, method='multigrate', name=''):

    if nmi_ or ari_:
        print('Clustering...')
        cluster_key = 'cluster'
        res_max, nmi_max, nmi_all = opt_louvain(adata,
                label_key=label_key, cluster_key=cluster_key, function=nmi,
                plot=False, verbose=False, inplace=True, force=True)

    results = {}
    # batch correction
    if asw_batch:
        print('ASW label/batch...')
        _, sil_clus = silhouette_batch(adata, batch_key=batch_key, group_key=label_key, embed=embed)
        results['ASW_label/batch'] = sil_clus.silhouette_score.mean()
    if pcr_batch:
        print('PCR batch...')
        results['PCR_batch'] = pcr_comparison(adata_old, adata, covariate=batch_key)
    if graph_connectivity_batch:
        print('Graph connectivity...')
        results['graph_conn'] = graph_connectivity(adata, label_key)

    # bio conservation
    if asw_label:
        print('ASW label...')
        results['ASW_label'] = silhouette(adata, group_key=label_key, embed=embed)
    if nmi_:
        print('NMI cluster/label...')
        results['NMI_cluster/label'] = nmi(adata, cluster_key, label_key)
    if ari_:
        print('ARI cluster/label...')
        results['ARI_cluster/label'] = ari(adata, cluster_key, label_key)
    if isolated_label_asw:
        print('Isolated label silhouette...')
        results['isolated_label_silhouette'] = isolated_labels(adata, label_key, batch_key, cluster=False, embed=embed)
    if isolated_label_f1:
        print('Isolated label F1...')
        results['isolated_label_F1'] = isolated_labels(adata, label_key, batch_key, cluster=True)

    df = pd.DataFrame.from_dict(results, orient='index', columns=['score'])
    df_return = df.copy()

    if save:
        df = df.transpose()
        df['method'] = method
        df['reference_time'] = 0
        df['query_time'] = 0
        df['data'] = name
        df.to_csv(save, index=False, sep=',')
    return df_return
