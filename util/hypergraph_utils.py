
import numpy as np

def construct_H_with_KNN_multi(X, K_neigs=[20],  is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    从原始节点特征矩阵启动多尺度超图顶点-边缘矩阵
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion 邻居扩张的数量
    :param split_diff_scale: whether split hyperedge group at different neighbor scale 是否在不同的邻居范围内分割超群组
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    H = []
    for i in range (len(X)):
        if type(K_neigs) == int:
            K_neigs = [K_neigs]
        dis_mat = Eu_dis(X[i])
        for k_neig in K_neigs:
            H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
            H.append(H_tmp)
    return H

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    从超图节点距离矩阵中构建超图发生率矩阵
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0] #3327
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    low_H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        farthest_idx = np.array(np.argsort(-dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        # 如果center_idx不在前k个中，将最后一位赋值为center_idx; 绝大部分存在，除非其他距离为0的很多
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                # dis_距离越小，计算结果越接近1;  按列进行构建超边
                    H[ center_idx,node_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[center_idx,node_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs, is_probH=False, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    从原始节点特征矩阵启动多尺度超图顶点-边缘矩阵
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion 邻居扩张的数量
    :param split_diff_scale: whether split hyperedge group at different neighbor scale 是否在不同的邻居范围内分割超群组
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """

    dis_mat = Eu_dis(X)
    H = construct_H_with_KNN_from_distance(dis_mat, K_neigs, is_probH, m_prob)
    return H
def generate_Lap_from_H(H):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_Lap_from_H(H)
    else:
        G = []
        for sub_H in H:
            G.append(generate_Lap_from_H(sub_H))
        return G


def _generate_Lap_from_H(H):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1) # 行和
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T
    G = np.eye(n_edge)-DV2 * H * W * invDE * HT * DV2
    return G

def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1) # Square of Euclidean norms of rows
    ab = x * x.T # Inner product of x with itself
    dist_mat = aa + aa.T - 2 * ab # Calculate squared Euclidean distances
    dist_mat[dist_mat < 0] = 0 # Set small negative values to zero (due to numerical precision)
    dist_mat = np.sqrt(dist_mat) # Take the square root to get Euclidean distances
    dist_mat = np.maximum(dist_mat, dist_mat.T)# Ensure the matrix is symmetric
    return dist_mat

