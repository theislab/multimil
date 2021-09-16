import torch
import time
from torch import nn
from torch.nn import functional as F
import numpy as np
import scanpy as sc
from operator import attrgetter
from itertools import cycle, zip_longest, groupby
from scipy import spatial
from .mlp import MLP

from .multivae_poe_cond import MultiVAE_PoE_cond, MultiVAETorch_PoE_cond
from ..datasets import SingleCellDatasetMIL

class Aggregator(nn.Module):
    def __init__(self,
                z_dim=None,
                scoring='sum',
                attn_dim=32 # D
                ):
        super(Aggregator, self).__init__()

        self.scoring = scoring

        if self.scoring == 'attn':
            self.attn_dim = attn_dim # attn dim from https://arxiv.org/pdf/1802.04712.pdf
            self.attention = nn.Sequential(
                nn.Linear(z_dim, self.attn_dim),
                nn.Tanh(),
                nn.Linear(self.attn_dim, 1, bias=False),
            )
        elif self.scoring == 'gated_attn':
            self.attn_dim = attn_dim
            self.attention_V = nn.Sequential(
                nn.Linear(z_dim, self.attn_dim),
                nn.Tanh()
            )

            self.attention_U = nn.Sequential(
                nn.Linear(z_dim, self.attn_dim),
                nn.Sigmoid()
            )

            self.attention_weights = nn.Linear(self.attn_dim, 1, bias=False)

    def forward(self, x):
        if self.scoring == 'sum':
            return torch.sum(x, dim=0) # z_dim
        elif self.scoring == 'attn':
            # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py (accessed 16.09.2021)
            self.A = self.attention(x)  # Nx1
            self.A = torch.transpose(self.A, 1, 0)  # 1xN
            self.A = F.softmax(self.A, dim=1)  # softmax over N
            return torch.mm(self.A, x).squeeze() # z_dim

        elif self.scoring == 'gated_attn':
            # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py (accessed 16.09.2021)
            A_V = self.attention_V(x)  # NxD
            A_U = self.attention_U(x)  # NxD
            self.A = self.attention_weights(A_V * A_U) # element wise multiplication # Nx1
            self.A = torch.transpose(self.A, 1, 0)  # 1xN
            self.A = F.softmax(self.A, dim=1)  # softmax over N
            return torch.mm(self.A, x).squeeze()  # z_dim

class MultiVAETorch_PoE_MIL(MultiVAETorch_PoE_cond):
    def __init__(self,
        encoders,
        decoders,
        shared_encoder,
        shared_decoder,
        mu,
        logvar,
        modality_vectors,
        device='cpu',
        condition=None,
        n_batch_labels=None,
        paired_dict={},
        modalities_per_group={},
        paired_networks_per_modality_pairs={},
        num_classes=None,
        scoring='attn',
        classifier_hiddens=[],
        normalization='layer',
        dropout=None,
        attn_dim=32
    ):
        super().__init__(
            encoders,
            decoders,
            shared_encoder,
            shared_decoder,
            mu,
            logvar,
            modality_vectors,
            device,
            condition,
            n_batch_labels,
            paired_dict,
            modalities_per_group,
            paired_networks_per_modality_pairs
        )

        z_dim = self.modality_vectors.weight.shape[1]

        if len(classifier_hiddens) > 0:
            classifier_hiddens.extend([classifier_hiddens[-1]]) # hack to make work with existing MLP module
            mil_dim = classifier_hiddens[-1]
        else:
            mil_dim = z_dim

        self.classifier = nn.Sequential(
                            MLP(z_dim, mil_dim, classifier_hiddens[:-1], output_activation='leakyrelu',
                                  dropout=dropout, norm=normalization, last_layer=False, regularize_last_layer=True),
                            Aggregator(mil_dim, scoring, attn_dim=attn_dim),
                            nn.Linear(mil_dim, num_classes)
                            )

        self = self.to(device)

    def get_params(self):
        params = super().get_params()
        params.extend(list(self.classifier.parameters()))
        return params

    def forward(self, xs, modalities, pair_groups, batch_labels, size_factors):
        hs = [self.to_shared_dim(x, mod, batch_label) for x, mod, batch_label in zip(xs, modalities, batch_labels)]
        zs = [self.bottleneck(h) for h in hs]
        mus = [z[1] for z in zs]
        logvars = [z[2] for z in zs]
        zs = [z[0] for z in zs]
        mus_joint, logvars_joint, _ = self.product_of_experts(mus, logvars, pair_groups)
        zs_joint = [self.reparameterize(mu_joint, logvar_joint) for mu_joint, logvar_joint in zip(mus_joint, logvars_joint)]
        out = self.prep_latent(xs, zs, zs_joint, modalities, pair_groups, batch_labels)
        zs = out[0]
        hs_dec = [self.z_to_h(z, mod) for z, mod in zip(zs, modalities)]
        rs = [self.decode_from_shared(h, mod, pair_group, batch_label) for h, mod, pair_group, batch_label in zip(hs_dec, modalities, pair_groups, batch_labels)]
        # classify
        predicted_scores = [self.classifier(z_joint) for z_joint in zs_joint]
        #if len(predicted_scores[0]) == 2:
    #        predicted_scores = [predicted_scores[i][0] for i in range(len(predicted_scores))]
        return rs, zs, mus+mus_joint, logvars+logvars_joint, pair_groups, modalities, batch_labels, xs, size_factors, predicted_scores

class MultiVAE_MIL(MultiVAE_PoE_cond):
    def __init__(
        self,
        adatas,
        names,
        pair_groups,
        condition=None,
        normalization='layer',
        z_dim=15,
        h_dim=32,
        hiddens=[],
        losses=[],
        output_activations=[],
        shared_hiddens=[],
        recon_coef=1,
        kl_coef=1e-4,
        integ_coef=1e-1,
        cycle_coef=0,
        dropout=0.2,
        device=None,
        loss_coefs=[],
        layers=[],
        theta=None,
        class_columns=None,
        bag_key=None,
        scoring='attn',
        classifier_hiddens=[]
    ):
        super().__init__(
        adatas,
        names,
        pair_groups,
        condition,
        normalization,
        z_dim,
        h_dim,
        hiddens,
        losses,
        output_activations,
        shared_hiddens,
        recon_coef,
        kl_coef,
        integ_coef,
        cycle_coef,
        dropout,
        device,
        loss_coefs,
        layers,
        theta
        )

        self.bag_key = bag_key
        self.class_columns = class_columns

        self.adatas, self.num_classes = self.reshape_adatas_mil(adatas, names, self.layers, pair_groups, self.batch_labels, class_columns, bag_key)

        self.model = MultiVAETorch_PoE_MIL(self.encoders, self.decoders, self.shared_encoder, self.shared_decoder,
                                   self.mu, self.logvar, self.modality_vecs, self.device, self.condition, self.n_batch_labels,
                                   self.pair_groups_dict, self.modalities_per_group, self.paired_networks_per_modality_pairs,
                                   self.num_classes, scoring, classifier_hiddens, normalization, dropout)

    def reshape_adatas_mil(self, adatas, names, layers, pair_groups, batch_labels, class_columns, bag_key):
        reshaped_adatas = {}
        classes = set()

        for modality, (adata_set, name_set, layer_set, pair_group_set, batch_label_set, class_column_set) in enumerate(zip(adatas, names, layers, pair_groups, batch_labels, class_columns)):
            for adata, name, layer, pair_group, batch_label, class_column in zip(adata_set, name_set, layer_set, pair_group_set, batch_label_set, class_column_set):
                adata.obs[bag_key] = adata.obs[bag_key].astype('category')
                adata.obs[class_column] = adata.obs[class_column].astype('category')

                for bag in adata.obs[bag_key].cat.categories:
                    adata_bag = adata[adata.obs[bag_key] == bag]

                    if len(adata_bag.obs[class_column].cat.categories) != 1:
                        raise ValueError(f'Classes not unique in the bag {name}_{bag}: {adata_bag.obs[class_column].cat.categories}')

                    class_ = adata_bag.obs[class_column].cat.categories[0]
                    classes.add(class_)

                    reshaped_adatas[name+'_'+bag] = {
                        'adata': adata_bag,
                        'layer': layer,
                        'modality': modality,
                        'pair_group': pair_group+'_'+bag,
                        'batch_label': batch_label,
                        'class': class_
                }
        return reshaped_adatas, len(classes)

    def make_datasets(self, adatas, val_split, modality_key, celltype_key, batch_size):
        train_datasets, val_datasets = [], []
        pair_group_train_masks = {}
        for name in adatas:
            adata = adatas[name]['adata']
            modality = adatas[name][modality_key]
            pair_group = adatas[name]['pair_group']
            batch_label = adatas[name]['batch_label']
            layer = adatas[name]['layer']
            class_ = adatas[name]['class']

            if pair_group in pair_group_train_masks:
                train_mask = pair_group_train_masks[pair_group]
            else:
                train_mask = np.zeros(len(adata), dtype=np.bool)
                train_size = int(len(adata) * (1 - val_split))
                train_mask[:train_size] = 1
                np.random.shuffle(train_mask)
                if pair_group is not None:
                    pair_group_train_masks[pair_group] = train_mask

            train_adata = adata[train_mask]
            val_adata = adata[~train_mask]

            train_dataset = SingleCellDatasetMIL(train_adata, name, modality, pair_group, celltype_key, batch_size, batch_label=batch_label, layer=layer, label=class_)
            val_dataset = SingleCellDatasetMIL(val_adata, name, modality, pair_group, celltype_key, batch_size, batch_label=batch_label, layer=layer, label=class_)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

        train_datasets = sorted(train_datasets, key=attrgetter('pair_group', 'modality'))
        val_datasets = sorted(val_datasets, key=attrgetter('pair_group', 'modality'))

        return train_datasets, val_datasets

    def train(
        self,
        n_iters=10000,
        batch_size=64,
        lr=3e-4,
        kl_anneal_iters=3000,
        val_split=0.1,
        modality_key='modality',
        celltype_key='cell_type',
        validate_every=1000,
        verbose=1,
        correct='all', # 'missing', 'none'
        version='2', # '1' or '2'
        kernel_type='gaussian',
        print_losses=False,
        ae_coef=1,
        classification_coef=1
    ):
        # configure training parameters
        print_every = n_iters // 50
        # create data loaders
        train_datasets, val_datasets = self.make_datasets(self.adatas, val_split, modality_key, celltype_key, batch_size)
        train_dataloaders = [d.loader for d in train_datasets]
        val_dataloaders = [d.loader for d in val_datasets]

        # create optimizers
        params = self.model.get_params()
        if self.theta is not None:
                params.extend([self.theta])
        optimizer_ae = torch.optim.Adam(params, lr)

        # the training loop
        epoch_time = 0  # epoch is the time between two consequtive validations
        self.model.train()
        for iteration, datas in enumerate(cycle(zip(*train_dataloaders))):
            tik = time.time()
            if iteration >= n_iters:
                break

            xs = [data[0].to(self.device) for data in datas]
            modalities = [data[2] for data in datas]
            pair_groups = [data[3] for data in datas]
            batch_labels = [data[-1] for data in datas]
            size_factors = [data[-2] for data in datas]
            labels = [data[-3] for data in datas]

            # forward propagation
            # TODO fix

            out = self.model(xs, modalities, pair_groups, batch_labels, size_factors)

            if len(out) == 5:
                rs, zs, mus, logvars, new_pair_groups = out
                zs_not_corrected = None
            elif len(out) == 9:
                rs, zs, mus, logvars, new_pair_groups, modalities, batch_labels, xs, size_factors = out
                zs_not_corrected = None
            elif len(out) == 10:
                rs, zs, mus, logvars, new_pair_groups, modalities, batch_labels, xs, size_factors, predicted_scores = out
                zs_not_corrected = None
            else:
                print('sth\'s wrong')
            losses = [self.losses[mod] for mod in modalities]

            loss_classification = self.calc_classification_loss(predicted_scores, labels)

            recon_loss, mse_loss, nb_loss, zinb_loss, bce_loss = self.calc_recon_loss(xs, rs, losses, batch_labels, size_factors)
            kl_loss = self.calc_kl_loss(mus, logvars)

            integ_loss = self.calc_integ_loss(zs, new_pair_groups, correct, kernel_type, version, zs_not_corrected)
            if self.cycle_coef > 0:
                cycle_loss = self.calc_cycle_loss(xs, zs, pair_groups, modalities, batch_labels, new_pair_groups, losses)
            else:
                cycle_loss = 0
            kl_coef = self.kl_anneal(iteration, kl_anneal_iters)  # KL annealing
            loss_ae = self.recon_coef * recon_loss + \
                      kl_coef * kl_loss + \
                      self.integ_coef * integ_loss + \
                      self.cycle_coef * cycle_loss


            loss_total = ae_coef * loss_ae + classification_coef * loss_classification
            # AE backpropagation
            optimizer_ae.zero_grad()
            loss_total.backward()
            optimizer_ae.step()

            # update progress bar
            if iteration % print_every == 0:
                # as function
                self._train_history['iteration'].append(iteration)
                self._train_history['loss'].append(loss_ae.detach().cpu().item())
                self._train_history['recon'].append(recon_loss.detach().cpu().item())
                self._train_history['class'].append(loss_classification.detach().cpu().item())
                for mod_loss, name in zip([mse_loss, nb_loss, zinb_loss, bce_loss], ['mse', 'nb', 'zinb', 'bce']):
                    name = 'recon_' + name
                    self._train_history[name].append(mod_loss.detach().cpu().item() if mod_loss !=0 else 0)
                self._train_history['kl'].append(kl_loss.detach().cpu().item())
                self._train_history['integ'].append(integ_loss.detach().cpu().item() if integ_loss != 0 else 0)
                self._train_history['cycle'].append(cycle_loss.detach().cpu().item() if cycle_loss != 0 else 0)

                if verbose >= 2:
                    self.print_progress_train(n_iters)

            # add this iteration to the epoch time
            epoch_time += time.time() - tik

            # validate
            if iteration % validate_every == 0 or iteration == n_iters - 1: # iteration > 0
                # as function
                # add average train losses of the elapsed epoch to the validation history
                self._val_history['iteration'].append(iteration)
                self._val_history['train_loss'].append(np.mean(self._train_history['loss'][-(validate_every//print_every):]))
                self._val_history['train_recon'].append(np.mean(self._train_history['recon'][-(validate_every//print_every):]))
                self._val_history['train_class'].append(np.mean(self._train_history['class'][-(validate_every//print_every):]))
                for mod_loss, name in zip([mse_loss, nb_loss, zinb_loss, bce_loss], ['mse', 'nb', 'zinb', 'bce']):
                    name_train = 'recon_' + name
                    name = 'train_recon_' + name
                    self._val_history[name].append(np.mean(self._train_history[name_train][-(validate_every//print_every):]))
                self._val_history['train_kl'].append(np.mean(self._train_history['kl'][-(validate_every//print_every):]))
                self._val_history['train_integ'].append(np.mean(self._train_history['integ'][-(validate_every//print_every):]))
                self._val_history['train_cycle'].append(np.mean(self._train_history['cycle'][-(validate_every//print_every):]))

                for i in range(self.n_modality):
                        name = 'mod_vec' + str(i) + '_norm'
                        self._val_history[name].append(torch.linalg.norm(self.model.modality_vectors.weight[i]).detach().cpu().item())
                        # cosine similarity
                        for j in range(i+1, self.n_modality):
                            name = f'cos_similarity_mod_vectors_{i}_{j}'
                            self._val_history[name] = 1-spatial.distance.cosine(self.model.modality_vectors.weight[i].detach().cpu().numpy(),
                                                                                self.model.modality_vectors.weight[j].detach().cpu().numpy())


                self.model.eval()
                self.validate(val_dataloaders, n_iters, epoch_time, kl_coef=kl_coef, verbose=verbose, correct=correct, kernel_type=kernel_type, version=version, ae_coef=ae_coef, classification_coef=classification_coef)
                self.model.train()
                epoch_time = 0  # reset epoch time

    def validate(self, val_dataloaders, n_iters, train_time=None, kl_coef=None, verbose=1, correct='all', kernel_type='gaussian', version='1', ae_coef=1, classification_coef=1):
        tik = time.time()
        val_n_iters = max([len(loader) for loader in val_dataloaders])

        if kl_coef is None:
            kl_coef = self.kl_coef

        # we want mean losses of all validation batches
        recon_loss = 0
        mse_l, nb_l, zinb_l, bce_l = 0, 0, 0, 0
        kl_loss = 0
        integ_loss = 0
        cycle_loss = 0
        class_loss = 0

        # for accuracy
        total = 0
        correct = 0


        for iteration, datas in enumerate(cycle(zip(*val_dataloaders))):
            # iterate until all of the dataloaders run out of data
            if iteration >= val_n_iters:
                break

            xs = [data[0].to(self.device) for data in datas]
            modalities = [data[2] for data in datas]
            pair_groups = [data[3] for data in datas]
            batch_labels = [data[-1] for data in datas]
            size_factors = [data[-2] for data in datas]
            labels = [data[-3] for data in datas]

            # forward propagation
            out = self.model(xs, modalities, pair_groups, batch_labels, size_factors)

            if len(out) == 5:
                rs, zs, mus, logvars, new_pair_groups = out
                zs_not_corrected = None
            elif len(out) == 9:
                rs, zs, mus, logvars, new_pair_groups, modalities, batch_labels, xs, size_factors = out
                zs_not_corrected = None
            elif len(out) == 10:
                rs, zs, mus, logvars, new_pair_groups, modalities, batch_labels, xs, size_factors, predicted_scores = out
                zs_not_corrected = None
            else:
                print('sth\'s wrong')

            losses = [self.losses[mod] for mod in modalities]

            # calculate the losses
            loss_classification = self.calc_classification_loss(predicted_scores, labels)

            true_labels = [labels[i] for i in range(0, len(labels), 2)]
            true_labels = torch.tensor(true_labels, dtype=torch.long).to(self.model.device)
            predicted_scores = torch.stack(predicted_scores)

            total += true_labels.size(0)
            _, predicted = torch.max(predicted_scores.data, 1)

            correct += (predicted == true_labels).sum().item()

            r_loss, mse_loss, nb_loss, zinb_loss, bce_loss = self.calc_recon_loss(xs, rs, losses, batch_labels, size_factors)

            class_loss += loss_classification
            recon_loss += r_loss
            mse_l += mse_loss
            nb_l += nb_loss
            zinb_l += zinb_loss
            bce_l += bce_loss
            kl_loss += self.calc_kl_loss(mus, logvars)
            integ_loss += self.calc_integ_loss(zs, new_pair_groups, correct, kernel_type, version, zs_not_corrected)
            if self.cycle_coef > 0:
                cycle_loss += self.calc_cycle_loss(xs, zs, pair_groups, modalities, batch_labels, new_pair_groups, losses)

        #print(f'\n Nastja <3 iter: {iteration}, corr: {correct}, tot: {total}')
        # calculate overal losses
        loss_ae = self.recon_coef * recon_loss + \
                  kl_coef * kl_loss + \
                  self.integ_coef * integ_loss + \
                  self.cycle_coef * cycle_loss

        loss_total = classification_coef * class_loss + ae_coef * loss_ae

        # logging
        self._val_history['val_loss'].append(loss_ae.detach().cpu().item() / val_n_iters)
        self._val_history['val_recon'].append(recon_loss.detach().cpu().item() / val_n_iters)
        self._val_history['val_class'].append(class_loss.detach().cpu().item() / val_n_iters)
        self._val_history['accuracy'].append(correct / total)

        for mod_loss, name in zip([mse_l, nb_l, zinb_l, bce_l], ['mse', 'nb', 'zinb', 'bce']):
            if mod_loss != 0:
                name = 'val_recon_' + name
                self._val_history[name].append(mod_loss.detach().cpu().item() / val_n_iters)

        self._val_history['val_kl'].append(kl_loss.detach().cpu().item() / val_n_iters)
        if integ_loss != 0:
            self._val_history['val_integ'].append(integ_loss.detach().cpu().item() / val_n_iters)
        else:
            self._val_history['val_integ'].append(0)
        if cycle_loss != 0:
            self._val_history['val_cycle'].append(cycle_loss.detach().cpu().item() / val_n_iters)
        else:
            self._val_history['val_cycle'].append(0)

        val_time = time.time() - tik
        if verbose == 1:
            self.print_progress_val(n_iters, train_time + val_time, end='')
        elif verbose >= 2:
            self.print_progress_val(n_iters, train_time + val_time, end='\n')

    def calc_classification_loss(self, predicted_scores, labels):
        true_labels = [labels[i] for i in range(0, len(labels), 2)]
        true_labels = torch.tensor(true_labels, dtype=torch.long).to(self.model.device)
        predicted_scores = torch.stack(predicted_scores)
        return F.cross_entropy(predicted_scores, true_labels)

    def test(self,
            adatas,
            names,
            pair_groups,
            modality_key='modality',
            celltype_key='cell_type',
            batch_size=64,
            batch_labels=None,
            layers=[],
            bag_key=None,
            class_columns=None
        ):

        if not bag_key:
            bag_key = self.bag_key

        if not class_columns:
            class_columns = self.class_columns

        if not batch_labels:
            batch_labels = self.batch_labels

        if len(layers) == 0:
            layers = [[None]*len(modality_adata) for i, modality_adata in enumerate(adatas)]

        pair_count = self.prep_paired_groups(pair_groups)

        # TODO: check if need unique_pairs_of_modalities
        self.model.paired_dict = self.pair_groups_dict
        #self.model.unique_pairs_of_modalities = self.unique_pairs_of_modalities
        self.model.modalities_per_group = self.modalities_per_group
        self.model.paired_networks_per_modality_pairs = self.paired_networks_per_modality_pairs

        adatas, _ = self.reshape_adatas_mil(adatas, names, layers, pair_groups=pair_groups, batch_labels=batch_labels, class_columns=class_columns, bag_key=bag_key)
        datasets, _ = self.make_datasets(adatas, val_split=0, modality_key=modality_key, celltype_key=celltype_key, batch_size=batch_size)
        dataloaders = [d.loader for d in datasets]

        ad_integrated = []
        classifier_adatas = [[] for _ in range(len(self.model.classifier[0].network))]

        with torch.no_grad():
            self.model.eval()

            for datas in zip_longest(*dataloaders):
                datas = [data for data in datas if data is not None]
                xs = [data[0].to(self.device) for data in datas]
                names = [data[1] for data in datas]
                modalities = [data[2] for data in datas]
                pair_groups = [data[3] for data in datas]
                celltypes = [data[4] for data in datas]
                indices = [data[5] for data in datas]
                batch_labels = [data[-1] for data in datas]
                labels = [data[-3] for data in datas]

                group_indices = {}
                for i, pair in enumerate(pair_groups):
                    group_indices[pair] = group_indices.get(pair, []) + [i]

                hs = [self.model.to_shared_dim(x, mod, batch_label) for x, mod, batch_label in zip(xs, modalities, batch_labels)]
                zs = [self.model.bottleneck(h) for h in hs]
                mus = [z[1] for z in zs]
                logvars = [z[2] for z in zs]
                zs = [z[0] for z in zs]
                mus_joint, logvars_joint, joint_pair_groups = self.model.product_of_experts(mus, logvars, pair_groups)
                zs_joint = [self.model.reparameterize(mu_joint, logvar_joint) for mu_joint, logvar_joint in zip(mus_joint, logvars_joint)]
                predicted_scores = [self.model.classifier(z_joint) for z_joint in zs_joint]

                # TODO: FIX
                A = []
                for z_joint in zs_joint:
                    self.model.classifier(z_joint)
                    A.append(self.model.classifier[-2].A.squeeze().detach().cpu().numpy())
                zs_corrected, new_modalities, new_pair_groups, new_batch_labels, xs, size_factors = self.model.prep_latent(xs, zs, zs_joint, modalities, pair_groups, batch_labels)

                true_labels = [labels[i] for i in range(0, len(labels), 2)]
                true_labels = torch.tensor(true_labels, dtype=torch.long).to(self.model.device)
                predicted_scores_stacked = torch.stack(predicted_scores)
                _, predicted_labels = torch.max(predicted_scores_stacked.data, 1)

                zs = zs_joint.copy()

                for i, layer in enumerate(self.model.classifier[0].network):
                    zs = [layer(z) for z in zs]
                    for z, pair in zip(zs, joint_pair_groups):
                        z = sc.AnnData(z.detach().cpu().numpy())
                        mods = np.array(names)[group_indices[pair], ]
                        z.obs['modality'] = '-'.join(mods)
                        z.obs['barcode'] = list(indices[group_indices[pair][0]])
                        z.obs['study'] = pair
                        z.obs[celltype_key] = celltypes[group_indices[pair][0]]

                        classifier_adatas[i].append(z)

                for i, (z_joint, pair, predicted_label, a) in enumerate(zip(zs_joint, joint_pair_groups, predicted_labels, A)):
                    z = sc.AnnData(z_joint.detach().cpu().numpy())
                    mods = np.array(names)[group_indices[pair], ]
                    z.obs['modality'] = '-'.join(mods)
                    z.obs['barcode'] = list(indices[group_indices[pair][0]])
                    z.obs['study'] = pair
                    z.obs[celltype_key] = celltypes[group_indices[pair][0]]
                    z.obs['predicted_label'] = predicted_label.detach().cpu().numpy()
                    z.obs['attn_score'] = a

                    ad_integrated.append(z)

        # concat for each classifier layer
        for i, adatas in enumerate(classifier_adatas):
            classifier_adatas[i] = sc.AnnData.concatenate(*classifier_adatas[i])

        return sc.AnnData.concatenate(*ad_integrated), classifier_adatas
