from sklearn.decomposition import PCA
import os, os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from utils_laj import movingavg
from utils_laj import cache
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


def plot_data(x, y, z, dim, labels_array, num_classes, save_fig, image_name):
    """
    Scatter plot for data distribution

    :param z:
    :param y:
    :param x:
    :param dim: dim=3 or dim=2, indicate the dimentions of the plot / type(dim)=int
    :param labels_array: an numpy array, which indicate the label for each and every data eg: [0 0 1 ... 2 2 2] / type(labels_array)=np.array
    :param num_classes: number of classes or clusters that the data belong / type(num_classes)=int
    :param save_fig: boolean value, save figure if True / type(save_fig)=bool
    :param image_name: string variable for plot title and file name in case its saved / type(image_name)=string

    """

    cmap = cm.get_cmap('gist_ncar', num_classes)
    if dim == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        sc = ax.scatter(x, y, z, c=labels_array.astype(np.float), edgecolor='k', cmap=cmap)
        plt.ylabel('y')
        plt.xlabel('x')
        plt.colorbar(sc, ticks=range(0, num_classes))
        plt.title(image_name)
        if save_fig == True:
            fig.savefig(image_name + '.png', bbox_inches='tight')
            plt.close(fig)
    elif dim == 2:
        fig = plt.figure()
        sc = plt.scatter(x, y, c=labels_array.astype(np.float), edgecolor='k', cmap=cmap)
        plt.ylabel('y')
        plt.xlabel('x')
        plt.colorbar(sc, ticks=range(0, num_classes))
        plt.title(image_name)
        if save_fig == True:
            fig.savefig(image_name + '.png', bbox_inches='tight')
            plt.close(fig)
    else:
        raise ValueError("invalid dimension valid either dim=2 or dim=3")


def clustering(Z, cluster_method, n_clusters=0, distance=0, min_samples=0):
    """
    Clustering function supports only kmeans and dbscan
    Eg:
    labels : [0 0 1 ... 2 2 2]
    event_cout :  [[  -1   72]
                  [   0  398]
                  [   1   26]
                  [   2 2520]]
    k_cls :  4

    :param Z: The data need to be clustered
    :param cluster_method: should be either 'kmeans' or 'dbscan' type(cluster_method)=string
    :param n_clusters: only for kmenas, type(n_clusters)=int
    :param distance:  only for dbscan type(distance)=float
    :param min_samples: only for dbscan type(min_samples)=float
    :return: 'labels' represents cluster labels for each and every points in 'Z' based on the 'clustering_method'
             'event_count[:,0]' column represents different cluster labels calculated by 'cluster_method'
             'k_cls' total number of different clusters exist in 'Z'
    """

    if cluster_method == 'kmeans':
        if isinstance(n_clusters, int) and n_clusters > 0:
            # kmeans = KMeans(n_clusters=k_cls,init='k-means++',n_init=k_cls)
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(Z)
            labels = kmeans.labels_
        else:
            raise ValueError("invalied 'n_clusters'")
    elif cluster_method == 'dbscan':
        if isinstance(min_samples, int) and min_samples > 0 and distance > 0:
            np.random.seed(42)
            db = DBSCAN(eps=distance, min_samples=min_samples).fit(Z)
            labels = db.labels_
        else:
            raise ValueError("invalied 'distance' or 'min_samples'")
    else:
        raise ValueError("Undefined cluster method, valied either cluster_method='kmeans' or cluster_method='dbscan'")

    unique, counts = np.unique(labels, return_counts=True)
    event_count = np.asarray((unique, counts)).T
    k_cls = len(unique)
    return labels, event_count, k_cls


def get_outlier_data(data_all, labels, cluster_method, event_count):
    """
    Hypothesis : outliers are the cluster, which containes smallest number of data points
    Since this implementation is focusing on removing anomalies and considering our data distribution,
    this hypothesis is almost accurate so far

    for dbscan, noise data lable (-1) and the label which belogs to the smallest cluster are identified separately

    :param data_all: raw data
    :param labels: cluster labels for each and every points in raw data, based on the 'cluster_method'
    :param cluster_method: should be either 'kmeans' or 'dbscan' type(cluster_method)=string
    :param event_count: an array which contains number of data points for specific cluster label type(event_count)=np.array, [None,2]
    :return: outlier data as np.array
    """

    if cluster_method == 'kmeans':
        smallest_cluster_label = np.argmin(event_count[:, 1])
        outliers_indexes = np.where(labels == smallest_cluster_label)[0]
        print('number of outlier datapoints : ', len(outliers_indexes), '\noutlier label : ', smallest_cluster_label)
        return data_all.iloc[outliers_indexes]
    elif cluster_method == 'dbscan':
        smallest_cluster_label = np.argmin(event_count[1:, 1])
        outliers_indexes = np.where(labels == smallest_cluster_label)[0]
        print('number of outlier datapoints : ', len(outliers_indexes), '\noutlier label : ', smallest_cluster_label)
        noise_indexes = np.where(labels == -1)[0]
        # print('number of noise indexes : ', len(noise_indexes))
        return data_all.iloc[outliers_indexes], data_all.iloc[noise_indexes]
    else:
        raise ValueError("Undefined cluster method, valid either cluster_method='kmeans' or cluster_method == 'dbscan'")


def get_pca_components(Z):
    """

    :param Z: accept normalized data, type(Z)=np.array shape(Z)=[n_samples, n_features]
    :return: pca values for respective components (x,y, and z)
    """
    pca = PCA(n_components=3)
    Z = pca.fit_transform(Z)
    x = Z[:, 0]
    y = Z[:, 1]
    z = Z[:, 2]
    return Z, x, y, z


def eigenvalue_analysis(data_std, save_fig, image_name):
    """
    plot covariance and correlation graphs
    plot cumulative explained variance and individual explained variance of eigen values

    :param data_std: data should be standardized type(data_std)=np.array, shape(data_std)=[features,samples]
    :param save_fig: whether to save the plot type(save_fig)=bool
    :param image_name: plot title and its saving file name incase its save_fig=True
    :return:
    """
    ############### covariance/correlation ########
    cov_mat = np.cov(data_std.T)
    # mean_vec = np.mean(data_ma_std, axis=0)
    # cov_mat = (data_ma_std - mean_vec).T.dot((data_ma_std - mean_vec)) / (data_ma_std.shape[0]-1)
    plt.figure()
    plt.imshow(cov_mat, label='covariance of data')
    plt.title('covariance of data')
    corr_mat = np.corrcoef(data_std.T)
    plt.figure()
    plt.imshow(corr_mat, label='correlation of data')
    plt.title('correlation of data')
    ##############################################

    ############### eigenvalues ##################
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    for ev in eig_vecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    print('\nEigenvalues \n%s' % eig_vals)

    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    with plt.style.context('seaborn-whitegrid'):
        fig = plt.figure()
        plt.bar(range(data_std.shape[1]), var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(range(data_std.shape[1]), cum_var_exp, where='mid',
                 label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        fig.savefig(image_name + '.png', bbox_inches='tight')
        plt.tight_layout()
    plt.show()
    ##############################################


def show_plots(save_fig):
    if not save_fig:
        plt.show()
