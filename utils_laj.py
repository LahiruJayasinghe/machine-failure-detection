import os
import pickle
import numpy as np

def cache(cache_path,obj=0):
    if os.path.exists(cache_path) and obj==0:
        with open(cache_path, mode='rb') as file:
            obj = pickle.load(file)
        print("- Data loaded from cache-file: " + cache_path)
    else:
        with open(cache_path, mode='wb') as file:
            pickle.dump(obj, file)
        print("- Data saved to cache-file: " + cache_path)

    return obj

def movingavg(data,window): #[n_samples, n_features]
    data = np.transpose(data)
    if data.ndim > 1 :
        tmp = []
        for i in range(data.shape[0]):
            ma = movingavg(np.squeeze(data[i]), window)
            tmp.append(ma)
        smas = np.array(tmp)
    else :
        w = np.repeat(1.0,window)/window
        smas = np.convolve(data,w,'valid')
    smas = np.transpose(smas)
    return smas #[n_samples, n_features]