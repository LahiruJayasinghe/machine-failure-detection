import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils_laj import cache

from procfunc import plot_data
from procfunc import clustering
from procfunc import get_outlier_data
from procfunc import get_pca_components
from procfunc import eigenvalue_analysis
from procfunc import show_plots

if __name__ == "__main__":

    # TODO: doc discription

    #### folder structure ##############################################################################################
    root = r'D:\LAHIRU\Work\KeySight\DataVisualization'
    data_folder = 'synchronized_data'
    anomaly_folder = 'first_round_anomaly_detection'
    ####################################################################################################################

    #### data parameters ###############################################################################################
    months = ['Nov']

    machines = ['192_168_28_28']
    # machines = ['hp36']

    discrete_sensors = True
    # discrete_sensors = True
    ####################################################################################################################

    #### clustering parameters #########################################################################################
    # cluster = 'dbscan'
    distance = 0.6
    min_samples = 20

    cluster = 'kmeans'
    n_clusters = 6
    ####################################################################################################################

    remove_outliers = False
    show_anomalies = False
    show_noise = False

    plot_eig_vals = False
    plot_events_based = False
    plot_cluster_based = True
    save_fig = False

    if discrete_sensors:
        file_tag = 'filtered_events_'
        pickle_tag = ''
    else:
        file_tag = 'filtered_events_sensors_'
        pickle_tag = 'SensorFiltered_'

    for month in months:
        for machine in machines:
            data_file = file_tag + month + '_' + machine + '.csv'
            data_fpath = os.path.join(data_folder, data_file)
            ftag = pickle_tag + month + '_' + machine
            path_half = os.path.join('pickle', ftag)
            print('data file: ', data_file, '\nfile tag: ', ftag)

            #### pre processing ########################################################################################
            # data_all = pd.read_csv(data_fpath, index_col='cf:timestamp')
            # data_all = cache(path_half+'_DataAll.pkl',data_all.values)
            # data_all = cache(path_half+'_DataAll.pkl')
            ############################################################################################################

            #### make events as classes ################################################################################
            # col = data_all.shape[1]
            # cls = data_all[:,col-1].astype(np.int32) # take events as classes
            # cls = cache(path_half + '_EventCls.pkl',cls)
            ############################################################################################################
            cls = cache(path_half + '_EventCls.pkl')

            #### evaluating number of events ###########################################################################
            unique, counts = np.unique(cls, return_counts=True)
            event_count = np.asarray((unique, counts)).T
            num_events = event_count[-1, 0]
            # print(event_count,num_events)
            ############################################################################################################

            #### pre processing ########################################################################################
            # data_all = data_all[:,0:col-1] # exclude events for PCA
            # print("excluding events", data_all.shape)

            # data_all = cache(path_half+'_ExcludeEvent.pkl',data_all)
            # data_all = cache(path_half+'_ExcludeEvent.pkl')

            # data_ma = movingavg(data_all,window=250)
            # data_ma = movingavg(data_all,window=1)
            # data_ma=data_all
            # print("data_ma shape : ",data_ma.shape)
            ############################################################################################################

            #### standerdization #######################################################################################
            # mean = np.mean(data_ma,axis=0)
            # data_ma_std = data_ma - mean
            # data_ma_std = StandardScaler().fit_transform(data_ma) #[n_samples, n_features]
            data_ma_std = cache(path_half + '_ExcludeEvent_normalized.pkl')
            ############################################################################################################

            if plot_eig_vals:
                eigenvalue_analysis(data_ma_std, save_fig, ftag + '_eigenvalues')

            #### show anomalies ########################################################################################
            if show_anomalies:
                for_anomaly = pd.read_csv(data_fpath, index_col='cf:timestamp')
                for_anomaly.plot().legend(loc='upper right')
                anomaly_fpath = os.path.join(anomaly_folder, ftag + '_outliers.csv')
                print(anomaly_fpath)
                anomaly_data = pd.read_csv(anomaly_fpath)
                anomaly_indexes = anomaly_data['index']
                for i in anomaly_indexes:
                    plt.axvspan(i, i + 1, color='green', alpha=0.3)
                    # plt.axvline(i, color='green', alpha=0.5)
                if show_noise:
                    noise_i = pd.read_csv(os.path.join(anomaly_folder, ftag + '_noise.csv'))['index']
                    for i in noise_i:
                        plt.axvspan(i, i + 1, color='blue', alpha=0.5)
                plt.show()
            ############################################################################################################

            data_all = pd.read_csv(data_fpath).reset_index()  # for anomaly analysis

            #### remove_outliers #######################################################################################
            if remove_outliers:
                if pickle_tag == 'SensorFiltered_':
                    max_index = 181759
                    min_index = 181435
                else:
                    max_index = 115694
                    min_index = 114428

                data_ma_std = np.delete(data_ma_std, slice(min_index, max_index), axis=0)
                cls = np.delete(cls, slice(min_index, max_index))
                data_all.drop(data_all.index[min_index:max_index], inplace=True)
            ############################################################################################################

            Z, x, y, z = get_pca_components(data_ma_std)  # perform pca on standardized data

            if cluster == 'kmeans':
                labels, event_count, k_cls = clustering(Z, cluster, n_clusters)
                outlier_data = get_outlier_data(data_all, labels, cluster, event_count)
                if save_fig:
                    outlier_data.to_csv(ftag + '_outliers.csv')
            elif cluster == 'dbscan':
                Z = Z[(Z[:, 0] > 5.5)]
                x = Z[:, 0]
                y = Z[:, 1]
                z = Z[:, 2]
                labels, event_count, k_cls = clustering(Z, cluster, distance=distance, min_samples=min_samples)
                outlier_data, noise_data = get_outlier_data(data_all, labels, cluster, event_count)
                if save_fig:
                    noise_data.to_csv(ftag + '_noise.csv')
                    outlier_data.to_csv(ftag + '_outliers.csv')
            print(event_count.T)

            if plot_events_based:
                plot_data(x, y, None, 3, cls, num_events, save_fig, 'first three Eigenvalues' + '_' + ftag)
                plot_data(x, y, None, 2, cls, num_events, save_fig, 'x-y' + '_' + ftag)
                plot_data(x, z, None, 2, cls, num_events, save_fig, 'x-z' + '_' + ftag)
                plot_data(y, z, None, 2, cls, num_events, save_fig, 'y-z' + '_' + ftag)

            if plot_cluster_based:
                plot_data(x, y, z, 3, labels, k_cls, save_fig, ftag + '_outliers_3d')
                plot_data(x, y, None, 2, labels, k_cls, save_fig, ftag + '_outliers_XY')
                plot_data(y, z, None, 2, labels, k_cls, save_fig, ftag + '_outliers_YZ')
                plot_data(x, z, None, 2, labels, k_cls, save_fig, ftag + '_outliers_XZ')

            show_plots(save_fig)
