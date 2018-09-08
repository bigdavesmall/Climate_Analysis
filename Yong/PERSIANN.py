import numpy as np
import pandas as pd
lead = 10
look = 5
def create_samples(path_data,path_Ganges,path_Brahmaputra,path_Meghna):
    persiann = np.load(path_data+'persiann.npy')
    timestamps = np.load(path_data+'time.npy')
    timestamps= pd.to_datetime(timestamps,format='%Y-%m-%d') 
    lat = np.load(path_data+'lat.npy')
    lon = np.load(path_data+'lon.npy')
    # get flow data 1983 - 2017
    target_Ganges = pd.read_csv(path_Ganges,index_col=3,header=0,parse_dates=True)
    idx  = (target_Ganges.index.year>1982)&(target_Ganges.index.year<2018)
    target_Ganges= target_Ganges.loc[target_Ganges.index[idx],'Q (m3/s)']
    
    target_Brahmaputra = pd.read_csv(path_Brahmaputra,index_col=3,header=0,parse_dates=True)
    idx  = (target_Brahmaputra.index.year>1982)&(target_Brahmaputra.index.year<2018)
    target_Brahmaputra= target_Brahmaputra.loc[target_Brahmaputra.index[idx],'Q (m3/s)']
    
    target_Meghna = pd.read_csv(path_Meghna,index_col=3,header=0,parse_dates=True)
    idx  = (target_Meghna.index.year>1982)&(target_Meghna.index.year<2018)
    target_Meghna= target_Meghna.loc[target_Meghna.index[idx],'Q (m3/s)']

    print ('persiann shape: ',persiann.shape, 'timestamps shape: ', timestamps.shape, 'latitude shape', lat.shape, 'longitude shape', lon.shape, 'target_Ganges shape', target_Ganges.shape, 'target_Brahmaputra shape', target_Brahmaputra.shape, 'target_Meghna shape', target_Meghna.shape)

    no_days = min(timestamps.shape[0],target_Ganges.shape[0],target_Brahmaputra.shape[0],target_Meghna.shape[0])
    Y_timestamps=[]
    X = np.zeros((no_days-lead-look,5,480,1440), float)
    Y_G = np.zeros((no_days-lead-look,1), float)
    Y_B = np.zeros((no_days-lead-look,1), float)
    Y_M = np.zeros((no_days-lead-look,1), float)
    for i, ts in enumerate(timestamps):
        if i+lead+look >no_days-1:
            break
        #use ith - i+look th days precipitation data to predict the flow of the day i+look+lead .
        X[i,:look,:,:]= persiann[i:i+look,:,:]
        Y_G[i] = target_Ganges[i+lead+look] 
        Y_B[i] = target_Brahmaputra[i+lead+look] 
        Y_M[i] = target_Meghna[i+lead+look] 
        Y_timestamps.append(timestamps[i+lead+look])
    Y_ts=pd.DataFrame(Y_timestamps, index=Y_timestamps,dtype='datetime64[ns]')
    # predict the flow of Jun, Jul, Aug, and Sep 
    idx  = (Y_ts.index.month > 5)&(Y_ts.index.month<10)
    X = X[idx,:,:,:]
    Y_G = Y_G[idx]
    Y_B = Y_B[idx]
    Y_M = Y_M[idx]
    
    Y_ts = Y_ts[idx]
    np.save('X',X)
    np.save('Y_G',Y_G)
    np.save('Y_B',Y_B)
    np.save('Y_M',Y_M)
    np.save('Y_ts',Y_ts)