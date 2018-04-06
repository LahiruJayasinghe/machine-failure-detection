import os, os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


def movingavg(data, window):
    w = np.repeat(1.0, window) / window
    smas = np.convolve(data, w, 'valid')
    return smas


root = r'D:\LAHIRU\Work\KeySight\OriginalData'

months = ['Sep']
machines = ['192_168_28_28', 'hp36']

for month in months:
    sensor_file = 'sensorData' + month + '.csv'
    event_file = 'machineEvent' + month + '.csv'

    SensorData_all = pd.read_csv(os.path.join(root, sensor_file), index_col='cf:timestamp',
                                 usecols=['cf:sensorType', 'cf:timestamp', 'cf:value', 'cf:equipmentId', 'cf:partId',
                                          'cf:moduleId'],
                                 dtype={'cf:sensorType': str, 'cf:timestamp': str, 'cf:value': str,
                                        'cf:equipmentId': str, 'cf:partId': str, 'cf:moduleId': str})

    SensorData_all.rename(columns={'cf:sensorType': 'sensorType', 'cf:timestamp': 'timestamp', 'cf:value': 'value',
                                   'cf:equipmentId': 'equipmentId', 'cf:partId': 'partId', 'cf:moduleId': 'moduleId'},
                          inplace=True)

    SensorData_all = SensorData_all.drop(SensorData_all[SensorData_all.sensorType == 'cf:sensorType'].index)

    SensorData_all.index = pd.to_datetime((SensorData_all.index).str[0:-6], format='%Y-%m-%dT%H:%M:%S.%f',
                                          errors='coerce')

    for machine in machines:
        eq_id = machine
        save_file = 'filtered_events_' + month + '_' + machine + '.csv'
        print(save_file)

        if eq_id == 'hp36':
            df_filterd = SensorData_all.loc[
                (SensorData_all['equipmentId'] == '192.168.28.252') | (SensorData_all['equipmentId'] == 'hp36')]
        elif eq_id == '192_168_28_28':
            df_filterd = SensorData_all.loc[SensorData_all['equipmentId'] == '192.168.28.28']  # filter by equipment ID

        df_filterd = SensorData_all

        df_filterd = df_filterd.loc[df_filterd.index.notnull()]
        df_filterd = df_filterd.loc[df_filterd.value.notnull()]
        df_filterd.fillna('Blank', inplace=True)

        print(df_filterd.equipmentId.unique())

        df_filterd.replace('BF1', 'BF', inplace=True)
        df_filterd.replace('VV1', 'VV', inplace=True)
        module_list = df_filterd.moduleId.unique()
        tmp = []
        for _module in module_list:
            df_moduled = df_filterd.loc[df_filterd['moduleId'] == _module]  # filter by module ID
            partId_list = df_moduled.partId.unique()
            print(_module, partId_list)
            for _part in partId_list:
                df_parted = df_moduled.loc[df_moduled['partId'] == _part]  # filter by part ID
                sensor_list = df_parted.sensorType.unique()
                # print(_part, sensor_list)
                for _sensor in sensor_list:
                    # if _sensor == 'Orientation X' or _sensor == 'Orientation Y' or _sensor == 'Orientation Z' or _sensor == '3.3V' or _sensor == '5.0V':
                    #     continue
                    # print(_sensor)
                    df_sensor = df_parted.loc[df_parted['sensorType'] == _sensor]  # filter by sensor type
                    # print(df_sensor.shape)
                    sensor_values = df_sensor[['value']]
                    sensor_values = sensor_values.loc[sensor_values.value.notnull()]
                    sensor_values['value'] = pd.to_numeric(sensor_values['value'], errors='coerce')
                    sensor_values = sensor_values[~sensor_values.index.duplicated(keep='first')]
                    sensor_values.sort_index(inplace=True)
                    sensor_values.rename(columns={'value': _module + "_" + _part + "_" + _sensor}, inplace=True)
                    # print(_sensor,'start',sensor_values.index[0],'end',sensor_values.index[-1])
                    tmp.append(sensor_values)
            print("***********")
        # print(len(tmp))

        ########################### Machine Evens #############################
        machine_event = pd.read_csv(os.path.join(root, event_file), index_col='cf:timestamp',
                                    dtype={'cf:eventType': str})

        machine_event.rename(columns={'Unnamed: 0': 'hashId', 'cf:eventType': 'event'}, inplace=True)
        machine_event.index = pd.to_datetime((machine_event.index).str[0:-6], format='%Y-%m-%dT%H:%M:%S.%f',
                                             errors='coerce')

        machine_event = machine_event.loc[machine_event.index.notnull()]

        hash_split = machine_event['hashId'].str.split(':', expand=True)
        _tmp = pd.concat([machine_event, hash_split], axis=1)
        event_data = _tmp.loc[:, ['event', 2]]
        event_data.rename(columns={2: 'equipmentId'}, inplace=True)

        # print("equipments : ",event_data.equipmentId.unique())

        print('before unique machine events : ', machine_event.event.unique())

        clearnup_nums = {'event': {'User input received': 1, 'Machine ready': 2, 'Machine alarm': 3,
                                   'Wait for user input': 4, 'Log out': 5, 'DUT all tests completed': 6,
                                   'DUT test time': 7,
                                   'DUT overall test result': 8, 'DUT serial number': 9, 'Board file loaded': 10,
                                   'Start running testplan': 11, 'Fixture locked': 12, 'Testplan loaded': 13,
                                   'Fixture unlocked': 14, 'Finish running diagnostic': 15, 'Machine busy/idle': 16,
                                   'Stop running testplan': 17, 'Start running diagnostic': 18}}  #
        event_data.replace(clearnup_nums, inplace=True)

        if eq_id == 'hp36':
            event_data_filtered = event_data.loc[(event_data['equipmentId'] == '192.168.28.252') | (
            event_data['equipmentId'] == 'hp36')]  # filter by equipment ID
        elif eq_id == '192_168_28_28':
            event_data_filtered = event_data.loc[event_data['equipmentId'] == '192.168.28.28']  # filter by equipment ID

        event_data_filtered = event_data_filtered[~event_data_filtered.index.duplicated(keep='first')]
        event_data_filtered.sort_index(inplace=True)
        event_values = event_data_filtered[['event']]
        # print('after unique machine events : ', event_values.event.unique())
        ######################################################################

        newindex = tmp[0].index.union(event_values.index)
        for i in range(1, len(tmp)):
            newindex = tmp[i].index.union(newindex)

        for i in range(0, len(tmp)):
            # print(tmp[i].shape)
            tmp[i] = tmp[i].reindex(newindex, method='nearest')
        event_values = event_values.reindex(newindex, method='nearest')
        print(tmp[0].shape)

        sync_results = pd.concat([tmp[0], tmp[1]], axis=1)
        for i in range(2, len(tmp)):
            sync_results = pd.concat([sync_results, tmp[i]], axis=1)
        sync_results = pd.concat([sync_results, event_values], axis=1)
        # print(list(sync_results))
        print('after unique machine events : ', sync_results.event.unique())
        unique, counts = np.unique(sync_results.event.values, return_counts=True)
        event_count = np.asarray((unique, counts)).T
        print(event_count)
        # show_col = ['Module 3_BF_Temperature Blower']
        # sync_results.plot(y=show_col)

        # print(sync_results.event.value_counts())
        # sync_results.to_csv(save_file)
        sync_results.plot()
        plt.show()
