#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 21:54:06 2018

@author: smalldave
"""

'''
    Deep Spatio-temporal Residual Networks
'''

from keras.layers import (
    Input,
    Activation,
    add,
    Dense,
    Reshape,
    Flatten,
    merge
)
import pandas as pd
import numpy as np
import os
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from netCDF4 import Dataset
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.utils.visualize_util import plot

def get_streamflow(dayN,day0):
    ganges = pd.read_csv('/media/smalldave/Storage/GBM/Ganges.csv')
    dates = (ganges.Year > 1984) & (ganges.Year<2018)
    ganges2 = ganges[dates]
    dates = (ganges2.Month>5) & (ganges2.Month<10)
    ganges2 = ganges2.loc[dates]
    ganges2 = ganges2.reset_index()
    frame = pd.DataFrame(ganges2['Q (m3/s)'])
    frame.columns = ['Q']
    for lag in np.arange(dayN,day0+1):
        x = ganges.loc[ganges2['index'] - lag, 'Q (m3/s)' ]
        x = pd.DataFrame(x)
        x.columns = [''.join(['Q_',str(lag)])]   
        x.index = frame.index
        frame = pd.concat([frame,x],axis=1)
    return frame,ganges2

def _shortcut(input, residual):
    print (input.shape)
    print (residual.shape)
    return add([input, residual])


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Conv2D(padding="same", strides=subsample, filters=nb_filter, kernel_size=(nb_row,nb_col))(activation)
    return f


def _residual_unit(nb_filter, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)
    return f


def ResUnits(residual_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter,
                                  init_subsample=init_subsample)(input)
        return input
    return f


def stresnet(c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32), t_conf=(3, 2, 32, 32), external_dim=8, nb_residual_unit=3):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    external_dim
    '''

    # main input
    main_inputs = []
    outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_flow, map_height, map_width = conf
            input = Input(shape=(nb_flow * len_seq, map_height, map_width))
            main_inputs.append(input)
            # Conv1
            conv1 = Conv2D (padding="same", filters=64, kernel_size=(3, 3))(input)
            # [nb_residual_unit] Residual Units
            residual_output = ResUnits(_residual_unit, nb_filter=64,
                              repetations=nb_residual_unit)(conv1)
            # Conv2
            activation = Activation('relu')(residual_output)
            conv2 = Conv2D(padding="same", filters=nb_flow, kernel_size=(3, 3))(activation)
            outputs.append(conv2)

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        from .iLayer import iLayer
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        main_output = add(new_outputs)

    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(output_dim=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(output_dim=nb_flow * map_height * map_width)(embedding)
        print(h1)
        activation = Activation('relu')(h1)
        external_output = Reshape((nb_flow, map_height, map_width))(activation)
        main_output = add([main_output, external_output])
    
    print('external_dim:', external_dim)

    #main_output = Activation('tanh')(main_output)
    flat = Flatten()(main_output)    
    flow = Dense(units=1)(flat)
    flow = Activation('relu')(flow)
    model = Model(inputs=main_inputs, outputs=flow)

    return model

lat0 = 17
lat1 = 32+8
lon0 = 70-8
lon1 = 101+8

filename='/media/smalldave/Storage/GBM/persiann_gfs_15day.nc'
infile=Dataset(filename,'r')
lat=list(infile.variables['lat'][:])
lon=list(infile.variables['lon'][:])

precip=infile.variables['precipitation'][:,:,lat.index(lat0):lat.index(lat1)+1,lon.index(lon0):lon.index(lon1)+1]


print(precip.shape)
frame,ganges2 = get_streamflow(15,20)

training = ganges2.Year < 2005
training_index = ganges2.loc[training].index

test = (ganges2.Year >2004) & (ganges2.Year<2017)
test_index = ganges2.loc[test].index

trainingFRAME = frame.loc[training_index]
testFRAME = frame.loc[test_index]
trainingPRECIP = precip[training_index,:,:,:]
testPRECIP = precip[test_index,:,:,:]
trainingQ = np.array(trainingFRAME['Q'])
testQ = np.array(testFRAME['Q'])
trainingFRAME.drop('Q',axis=1,inplace=True)
testFRAME.drop('Q',axis=1,inplace=True)

trainingFRAME = np.array(trainingFRAME)
testFRAME = np.array(testFRAME)

time,fhour,lat_,lon_ = np.shape(trainingPRECIP)
nb_residual_unit = 16
nb_epoch = 500
batch_size = 32

c_conf = (fhour,1,lat_,lon_)
_,external_dim = np.shape(trainingFRAME) 
#external_dim = 0
lr = 0.0002
hyperparams_name = 'c{}.resunit{}.lr{}'.format(21, nb_residual_unit, lr)
fname_param = "/media/smalldave/Storage/GBM/best_parameters.hdf5"

early_stopping = EarlyStopping(monitor='mean_squared_error', patience=10, mode='min')
model_checkpoint = ModelCheckpoint(fname_param, verbose=0, save_best_only=True, mode='min')

model = stresnet(c_conf=c_conf, p_conf=None, t_conf=None,
                 external_dim=external_dim, nb_residual_unit=nb_residual_unit)
#   
 
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

print(model.summary())
#
Xtrain = [trainingPRECIP,trainingFRAME]
#Xtrain = trainingPRECIP

Xtest = [testPRECIP,testFRAME]
#Xtest = testPRECIP
history = model.fit(Xtrain, trainingQ,
                epochs=nb_epoch,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[early_stopping,model_checkpoint],
                verbose=1)
#    
#model.save_weights(os.path.join('MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
#pickle.dump((history.history), open(os.path.join(path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
#
#model.load_weights(fname_param)
score = model.evaluate(Xtrain, trainingQ, batch_size=trainingQ.shape[0] // 48, verbose=0)
print('Train score: %.6f rmse (norm): %.6f' %
      (score[0], score[1]))

score = model.evaluate(Xtest, testQ, batch_size=testQ.shape[0], verbose=0)
print('Test score: %.6f rmse (norm): %.6f' %
      (score[0], score[1]))

Qhat = model.predict(Xtest, batch_size=testQ.shape[0], verbose=0)
Q=pd.concat([pd.DataFrame(Qhat),pd.DataFrame(testQ)],axis=1)
Q.columns = ['Predicted','Observed']
