#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 局部异常因子算法是一种通过计算给定样本相对于其邻域的局部密度偏差来实现异常检测的算法。
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd


# min/max 归一化处理
def Min_Max_Normal(x):
    return (x - x.min()) / x.max() # 计算异常得分

# 计算每个簇中样本的数量
def compute_clusters_sample_size(km_num=0, km_lable=0):
    cluster_sizes = []
    for label in range(km_num):
        size = sum(km_lable == label)
        cluster_sizes.append(size)
    return cluster_sizes

def compute_large_and_small_clusters(df_cluster_sizes, num_point_in_large_clusters, beta):
    """
    划分大簇集合和小簇集合
    条件1:大簇的样本总数为总样本数的0.9
    条件2:最小大簇的样本数大于或等于最大小簇的样本数的5倍
    :param df_cluster_sizes: 每个簇的样本数量
    :param num_point_in_large_clusters: 大簇的样本数量
    :param beta: 阈值            
    :return: 大簇集合，小簇集合
    """
    large_clusters, small_clusters = [], [] # 大簇集合，小簇集合
    sizes = df_cluster_sizes['size'].values
    clusters = df_cluster_sizes['cluster'].values
    n_clusters, found_b, count = len(clusters), False, 0
    for i in range(n_clusters):
        satisfy_alpha, satisfy_beta = False, False
        if found_b:
            small_clusters.append(clusters[i])
            continue
        count += sizes[i]

        # 划分大簇和小簇
        raise NotImplementedError('编写划分条件以及划分过程')
        
    return large_clusters, small_clusters

# 计算两点间的距离
def get_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# 计算样本点的异常得分
#   一个点到最近的大簇的距离作为异常得分，如果这个点是大簇里面的点，那么直接计算他到簇中心的距离即可；
#   如果这个点不是大簇中的点，那么就要分别计算其到所有大簇的距离，选最小的那个作为异常得分。
def decision_function(X, labels, large_clusters, large_cluster_centers, km):
    n, distances = len(labels), []
    for i in range(n):
        p, label = X[i], labels[i]
        if label in large_clusters:
            center = km.cluster_centers_[label]
            d = get_distance(p, center)
        else:
            d = None
            for center in large_cluster_centers:
                d_temp = get_distance(p, center)
                if d is None:
                    d = d_temp
                elif d_temp < d:
                    d = d_temp
        distances.append(d)
    distances = np.array(distances)
    return distances


# 待测试程序
def solution():
    #读取并处理样本数据
    df = pd.read_excel("./Sample - Superstore.xls")
    num_point = df.shape[0]
    x1, y1 = df['Sales'].values, df['Profit'].values
    x, y = Min_Max_Normal(x1), Min_Max_Normal(y1)
    X = [[a, b] for (a, b) in zip(x, y)]
    X = np.array(X)
    
    #设置超参数
    alpha, beta = 0.9, 5

    # 设置大小簇分解，即大簇总数量为样本总数量的0.9
    num_point_in_large_clusters = int(num_point * alpha)

    # 用k-means算法对样本进行聚类，得到8个簇集合
    km = KMeans(n_clusters=8)
    km.fit(X)
    
    #得到每个簇对应的样本数集合
    cluster_sizes = compute_clusters_sample_size(
        km.n_clusters, km.labels_)
     
    #根据簇中的样本数量进行降序排序
    df_cluster_sizes = pd.DataFrame()
    df_cluster_sizes['cluster'] = list(range(8))
    df_cluster_sizes['size'] = df_cluster_sizes['cluster'].apply(
        lambda c: cluster_sizes[c])
    df_cluster_sizes.sort_values(by=['size'], ascending=False, \
        inplace=True)
    
    #将排序后的簇集合划分为大簇集合和小簇集合
    large_clusters, small_clusters = compute_large_and_small_clusters(
        df_cluster_sizes, num_point_in_large_clusters, beta)
        
    #计算大簇集合中每个簇的聚类中心
    large_cluster_centers = km.cluster_centers_[large_clusters]
    
    #计算每个样本数据的异常得分
    distances = decision_function(X, km.labels_, large_clusters, large_cluster_centers, km)
    
    #设置异常数据比例为1%
    threshold = np.percentile(distances, 99)
    anomaly_labels = (distances > threshold) * 1

    # 返回前5个异常样本点的编号
    result = []
    for a, b, c in zip(x[anomaly_labels == 1][:5], \
        y[anomaly_labels == 1][:5],
                       distances[:5]):
        result.append((a, b))
    return result


if __name__ == '__main__':
    solution()
    pass
