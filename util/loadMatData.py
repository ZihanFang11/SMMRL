
import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
import h5py
from sklearn.preprocessing import normalize
import random

def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict


def generate_partition(labels, ratio,seed=20):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {}  ## number of labeled samples for each class
    total_num = round(ratio * len(labels))
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1)  # min is 1

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    index = [i for i in range(len(labels))]
    # print(index)
    if seed >= 0:
        random.seed(seed)
        random.shuffle(index)
    labels = labels[index]
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            p_labeled.append(idx)
            total_num -= 1
        else:
            p_unlabeled.append(idx)
    return p_labeled, p_unlabeled


def load_data_semi(args, dataset, normlize=True):
    feature_list = []
    if dataset == "AwA":
        data = h5py.File(args.path + dataset + '.mat')
        features = data['X']
        for i in range(features.shape[1]):
            if normlize:
                feature_list.append(normalize(data[features[0][i]][:].transpose()))
        labels = data['Y'][:].flatten()
    else:
        data = sio.loadmat(args.path + dataset + '.mat')
        features = data['X']
        for i in range(features.shape[1]):
            if normlize:
                features[0][i] = normalize(features[0][i])
            feature = features[0][i]
            if ss.isspmatrix_csr(feature):
                feature = feature.todense()
                print("sparse")
            # feature = torch.from_numpy(feature).float().to(args.device)
            feature_list.append(feature)
        labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    idx_labeled, idx_unlabeled = generate_partition(labels=labels, ratio=args.train_ratio,seed=-1)


    labels = torch.from_numpy(labels).long()

    return feature_list, labels,idx_labeled, idx_unlabeled
def generate_partition_ind(labels,ind, ratio=0.1):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {}  ## number of labeled samples for each class
    total_num = round(ratio * len(labels))
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1)  # min is 1

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            p_labeled.append(ind[idx])
            total_num -= 1
        else:
            p_unlabeled.append(ind[idx])
    return p_labeled, p_unlabeled


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


