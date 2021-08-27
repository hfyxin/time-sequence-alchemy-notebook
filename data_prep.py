# supporting classes and functions for preparing engine diagnostic data.
import os
import scipy.io
import pandas as pd
import numpy as np
from scipy import signal
import librosa
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

def get_filenames(folder, extension=None, joinpath=True):
    '''
    Return a list of filenames in given folder.

    Parameters:
    -----------
    folder: string.
    extension: string. If specified, only files with same extension will be returned.
    joinpath: If False, return only filenames. If True, return file paths.

    Returns:
    --------
    a list of file names in string.

    '''

    # obtain all files in a folder
    file_list = os.listdir(folder)

    # filter by extension
    if extension:
        file_list = [fn for fn in file_list if fn.endswith(extension)]

    # join path
    if joinpath:
        file_list = [os.path.join(folder,fn) for fn in file_list]

    return file_list


def load_and_parse(file_name):
    '''Load .mat file, parse it and covert to DataFrame. 
    Signal data have 2 different sampling rate. Return 2 DataFrames.
    
    Parameters:
    -----------
    file_name: string, only accept files with .mat extension.

    Returns:
    --------
    DataFrames containing cycle, crank angle and sensor values. 
    Sensor with the same sampling rate are grouped in one DataFrame.
    '''

    if file_name.endswith('.mat'):
        # parse .mat data
        # sampling rate: signal_1 - 7200 per cycle, signal_2 - 719 per cycle.
        signal_1, signal_2 = parse_mat(scipy.io.loadmat(file_name))

        # convert to DataFrame
        signal_1 = pd.DataFrame(signal_1)
        signal_2 = pd.DataFrame(signal_2)
        
        return signal_1, signal_2

    else:
        return False


def feature_selection(df1, df2, features):
    '''Select the features in signal DataFrame to keep. 
    Must be done right after .mat is imported as DataFrame.
    df1, df2 are two DataFrames for high & low res signals.'''

    # cycles and angles are kept by default
    if 'angles' not in features:
        features = ['angles'] + features
    if 'cycles' not in features:
        features = ['cycles'] + features

    # for signal 1
    columns = list_intersect(features, df1.columns)
    df1 = df1.loc[:, columns]
    # for signal 2
    columns = list_intersect(features, df2.columns)
    df2 = df2.loc[:, columns]

    return df1, df2


def list_intersect(lst1, lst2): 
    '''Find the intersection of two lists.'''
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 


# Extract data from the loaded mat file
def parse_mat(mat):
    '''Parse the .mat data imported by scipy.
    
    Parameters:
    -----------
    mat: dict. Raw Matlab data file loaded by scipy. May contain these keys:
     - 'CAO': sensor data in crank angle domain.
     - 'TM': sensor data in time domain.
     - 'ATR': crank angle vs timestamp.
     - 'PAR': test parameters, unstructured.

    Returns:
    --------
    Selected sensor values in a dict. 
    Sensors with same sampling rate are grouped in one dict.
    '''

    # High resolution sensor data
    # 4 knock sensors, 2 accelerometers, 1 microphone, 1 pressure sensor
    # 7200 data points per cycle, in crank angle domain (CAO)
    # The name CAO comes from original .mat file.
    CAO = {}
    CAO['cycles'] = mat['CAO']['UNI0'][0][0]['Cycles'].flat[0]
    CAO['angles'] = mat['CAO']['UNI0'][0][0]['Angles'].flat[0]
    CAO['knock1'] = mat['CAO']['UNI0'][0][0]['Values'].flat[0]
    CAO['knock2'] = mat['CAO']['UNI_A0'][0][0]['Values'].flat[0]
    CAO['knock3'] = mat['CAO']['UNI_B0'][0][0]['Values'].flat[0]
    CAO['knock4'] = mat['CAO']['UNI_C0'][0][0]['Values'].flat[0]
    CAO['hall']   = mat['CAO']['UNI_D0'][0][0]['Values'].flat[0]  # crank angle position
    CAO['acc1']   = mat['CAO']['UNI_E0'][0][0]['Values'].flat[0]
    CAO['acc2']   = mat['CAO']['UNI_F0'][0][0]['Values'].flat[0]
    CAO['pres']   = mat['CAO']['PCYL1'][0][0]['Values'].flat[0]

    # High resolution engine speed data
    # 719 data points per cycle, in crank angle domain (CAO)
    CAO_spd = {}
    CAO_spd['cycles'] = mat['CAO']['SPEED_HIRES0'][0][0]['Cycles'].flat[0]
    CAO_spd['angles'] = mat['CAO']['SPEED_HIRES0'][0][0]['Angles'].flat[0]
    CAO_spd['spd']    = mat['CAO']['SPEED_HIRES0'][0][0]['Values'].flat[0]
    
    # parameters. There are 27 items, pick the ones useful
    # PAR = mat['PAR'][0,0]
    
    # crank angle, timestamp, cycle count
    # ATR = mat['ATR'][0,0]
    
    # TM has same signals as CAO, but in time domain (TM)
    # TM = mat['TM'][0,0][i][0,0]

    # Adjust the shape of each data, (n,1) to (n,)
    for key, value in CAO.items():
        CAO[key] = value.squeeze(axis=1)
    for key, value in CAO_spd.items():
        CAO_spd[key] = value.squeeze(axis=1)
    
    return CAO, CAO_spd


def create_sample_by_cycle(df1, df2, cycle_per_sample=1, overlap=False):
    '''Create data samples from a whole DataFrame imported from .mat file.

    Parameters:
    -----------
    df1: the DataFrame that contains 201 engine cycles. 
    Must have these columns:['cycles', 'angles', 'at least 1 sensor']

    df2: same as df1, sensors at different sampling ratio

    cycle_per_sample: how many engine cycles does each output sample contain.

    overlap: does the sample overlap with each other. (like a sliding window)

    Returns:
    --------
    two lists of DataFrames, each contains one or more engine cycles.
    '''

    df1_list = []
    df2_list = []

    n_cycles = df1['cycles'].max() + 1  # 201 cycles

    sample_list = []  # should have 2 values, sensor_high, sensor_low

    # cycle_per_sample not implemented yet.
    # if do so, pad_engspd() will need to change
    if cycle_per_sample == 1: pass
    else: pass
    
    # separate data by cycle    
    for i in range(n_cycles): 
        df1_list.append(df1.loc[df1['cycles'] == i])
        df2_list.append(df2.loc[df2['cycles'] == i])
    
    # drop the useless columns

    # Fix engine speed missing value
    if 'spd' in df1.columns:
        df1_list = [pad_engspd(df) for df in df1_list]
    elif 'spd' in df2.columns:
        df2_list = [pad_engspd(df) for df in df2_list]
    else:
        pass

    return df1_list, df2_list


def pad_engspd(data_df):
    '''Engine speed data comes from Crankshaft position sensor, which has a missing peak,
    giving 719 points per cycle. Pad it to 720.

    Parameters:
    -----------
    data_df: the DataFrame contains 'angles' and 'spd' data, 1 cycle only.

    Return:
    -------
    DataFrame that added 1 row in the end.
    '''
    # get the last value of each column
    if 'angles' in data_df.columns:
        angles_val = data_df['angles'][-1:].values[0]
        angles_val  = angles_val + 1     # 358.5 + 1 = 359.5
    else:
        print('Cannot find angles column in the data.')
        raise
        
    if 'spd' in data_df.columns:
        spd_val = data_df['spd'][-1:].values[0]
    else:
        print('Cannot find spd column in the data.')
        raise
    
    # pad the value
    pad_value = {
        'angles': angles_val,
        'spd': spd_val,
    }

    return data_df.append(pad_value, ignore_index=True)


def moving_average(df1_list, df2_list, MA_window):
    '''Apply moving average to every DataFrames in the list.'''
    
    if MA_window: print("Moving average not implemented.")
    return df1_list, df2_list


def create_scalers(data_df, MA_window, scalers={}):
    '''Create or update scalers for engine diagnostic data.
    
    Parameters:
    -----------
    MA_window: the moving average window used to process data.

    scalers: list of Sklearn scaler objects. if {}, create new scalers.
    if specified, update given scalers.

    Returns:
    --------
    scalers: the updated/created scalers objects.
    '''
    
    # create new ones if they don't exist
    if scalers == {}:
        scalers['knock'] = StandardScaler()
        scalers['pres'] = MinMaxScaler(feature_range=(0,1))
        scalers['acc'] = StandardScaler()
        scalers['hall'] = MinMaxScaler(feature_range=(0,1))  # for the peak values of mic
    
    # fit scaler objects
    MA_window = max(1, int(MA_window/8))  # divide by 8 to prevent excessive pos/neg cancellation
    # knock sensor
    scalers['knock'].partial_fit(data_df.loc[:,['knock1','knock2','knock3','knock4']].rolling(window=MA_window).mean())
    # accelerometer 2
    scalers['acc'].partial_fit(data_df.loc[:,['acc1', 'acc2']].rolling(window=MA_window).mean())  # window size same as knock sensor
    # pressure signal
    scalers['pres'].partial_fit(data_df.loc[:,['pres']])  
    # accelerometer 1, fit the spikes
    x = data_df['hall'].values
    _, properties = find_valleys(x, height=2)
    scalers['hall'].partial_fit(properties['peak_heights'].reshape(-1,1))

    return scalers


def create_scalers_lo(data_df, MA_window, scalers={}):
    '''Create or update scalers for low-res data, i.e., engine speed data.
    
    Parameters:
    -----------
    MA_window: the moving average window used to process data.
    engine speed typically don't need MA, can put an arbitrary number just for safety.

    scalers: list of Sklearn scaler objects. if {}, create new scalers.
    if specified, update given scalers.

    Returns:
    --------
    scalers: the updated/created scalers objects.
    '''

    # create new ones if they don't exist
    if scalers == {}:
        scalers['spd'] = MinMaxScaler(feature_range=(0,1))
    
    # fit scaler objects: engine speed
    MA_window = max(1, MA_window)  # 
    scalers['spd'].partial_fit(data_df.loc[:,['spd']].rolling(window=MA_window).mean())

    return scalers


class SequenceScaler(BaseEstimator, TransformerMixin):
    '''
    Select an appropriate scaler for each sequence feature.
    Can specify scalers explicitly, otherwiser will automatically decide btw
    MinMax and Standard scaler.
    '''
    def __init__(self, scalers=[]):
        # placeholder for a number of various scalers
        self.scalers = scalers
        if scalers == []:
            self.auto_scaler = True
        else:
            self.auto_scaler = False
    
    def fit(self, X, y=None):
        ''' X should be 3d: (batch, seq_length, feature). Fit one scaler for each feature.'''
        
        for i in range(X.shape[-1]):
            x_row = X[:,:,i].flatten()
            x_row = np.expand_dims(x_row, axis=1)

            # automatically choose the a scaler
            if self.auto_scaler:
                if x_row.min() < -0.01:
                    self.scalers.append(StandardScaler())
                else:
                    self.scalers.append(MinMaxScaler(feature_range=(0,1)))
            # use pre-set scaler
            else:
                pass

            # fit the scaler
            self.scalers[i].fit(x_row)

        return self
    
    def transform(self, X):
        '''Transform sequence batches, using the explicit way since sklearn's scaler applies to 1d only.
        Note that the input is changed.
        '''
        X_new = X.copy()
        # Apply scaler on each dimension, explicitly
        for i, scaler in enumerate(self.scalers):
            if type(scaler) == StandardScaler:
                X_new[:,:,i] = (X[:,:,i] - scaler.mean_) / scaler.scale_
            elif type(scaler) == MinMaxScaler:
                X_new[:,:,i] = (X[:,:,i] - scaler.data_min_) / scaler.data_range_

        return X_new


class ChromaScaler(BaseEstimator, TransformerMixin):
    '''
    a modified Min-Max scaler that applies to a multi dimensional array.
    Since the input data can be both positive and negative, the scaling won't change their signs.

    Fitting: scale k = (max - min) / 2
    Transform: output = input / k 
    '''

    def __init__(self):
        '''.'''
        self.data_max = 0
        self.data_min = 0
        self.scale_k = 0
    
    def fit(self, X, y=None):
        ''' Fit one scaler for all data.'''
        
        self.data_max = X.max()
        self.data_min = X.min()

        self.scale_k = (self.data_max - self.data_min) / 2
        return self
    
    def transform(self, X):
        '''.'''
        return X / self.scale_k


def sync_by_resample(df1_list, df2_list, resample_rate):
    '''Merge df1 and df2 by synchronise them at given resample_rate.

    Parameters:
    -----------
    df1_list: the DataFrame contains high resolution signals at 7200 points/cycle

    df2_list: the DataFrame contains low resolution signals at 720 points/cycle

    Returns:
    --------
    one numpy array that only keeps sensor values, discard other columns.
    '''

    # placeholder
    array1 = []
    array2 = []

    # get the values column, discard the rest
    df1_val_col = df1_list[0].columns.tolist()
    df2_val_col = df2_list[0].columns.tolist()
    
    df1_val_col.remove('angles')
    df1_val_col.remove('cycles')
    df2_val_col.remove('angles')
    df2_val_col.remove('cycles')

    for df1, df2 in zip(df1_list, df2_list):
        array1.append(signal.resample(df1[df1_val_col], resample_rate))
        array2.append(signal.resample(df2[df2_val_col], resample_rate))
    
    array1 = np.stack(array1)
    array2 = np.stack(array2)
    
    return np.concatenate((array1, array2), axis=2)


def resample_to_OBD(df, df_spd, rate=100):
    '''Resample the input sigal to a lower rate to simulate OBD output.
    E.g. given a rate of 100Hz, count the accumulated angle. Once the counter hit 10ms threshold,
    note down the values at the point. 
    This will be able to adapt to different engine speed.

    Parameters:
    -----------
    df: DataFrame of high resolution sensor data.

    df_spd: DataFrame of low resolution sensor data, should only contain engine speed.

    rate: resample rate for OBD, Hz.

    Returns:
    --------
    DataFrame containing all high speed sensors, speed, and timestamp (ts). at the given sampling rate.
    timestamps are only used to check validity, no real use.

    Note this function uses a better coding appraoch than the sync_by_resample().
    '''

    df = merge_signals(df, df_spd)

    # Calculate accumulated timestamps, and resample points.
    ## Hard-core Algorithms Alert! ##

    # extract numpy values
    deg = df.angles.values
    cycle = df.cycles.values
    spd = df.spd.values

    # angle difference for each point (deg)
    deg = cycle * 720 + deg
    delta_deg = np.zeros(deg.shape)
    delta_deg[1:] = deg[1:] - deg[0:-1]

    # time difference calculated from angle difference (s)
    delta_tm = delta_deg / spd / 360

    # accumulated time (s)
    accu_tm = np.cumsum(delta_tm)

    # accumulated time rounded (floor) to int
    ts_rounded = accu_tm * rate
    ts_rounded = np.floor(ts_rounded).astype('int')

    # resample point: pick the first change of rounded time
    rs_point = np.zeros(ts_rounded.shape)
    rs_point[1:] = ts_rounded[1:] - ts_rounded[0:-1]
    rs_point = rs_point.astype('bool')

    # add the new values to dataframe
    df['ts'] = accu_tm
    df['rs_point'] = rs_point

    return df.loc[df.rs_point==True]


def merge_signals(df_high, df_low):
    '''Merge two set of signals with different sampling rate.
    The method is smiliar to SQL's OUTER JOIN.
    df_low's cycles/angles stampes must appear in df_high.
    '''

    n_total = df_high.shape[0]
    
    # unify the values
    df_high['angles'] = df_high['angles'].round(2)   
    df_low['angles'] = df_low['angles'].round(2)
    
    # other columns should have already uniformed

    # FULL OUTER JOIN
    df = pd.merge(df_high, df_low, on=['cycles', 'angles'], how='outer')

    # check if merge is successful
    assert df.shape[0] == n_total, 'Merge of two signal DataFrame failed.'

    # fill the empty speed data using zoh
    df['spd'] = df['spd'].fillna(method='ffill')
    df['spd'] = df['spd'].fillna(method='bfill') # fill the few points at beginning

    return df


def crank_to_time_single(df, df_engspd, sample_rate_rev=3600):
    '''Convert signals from crank (angle) domain to time domain for 1 sample,
    assuming within this sample, the engine speed won't change.

    Parameters:
    -----------
    df: the DataFrames that contain signals to be processed.

    df_engspd: the DataFrame that (only) contain engine speed.

    sample_rate_rev: sample rate per revolution in crank angle domain. 
    In CMHT engine test is 3600 rev-1, ie. peroid is 0.1 degree. Unlikely to change.

    Returns:
    --------
    a numpy array of coverted data, sampling rate (Hz), and engine speed (rpm).
    '''
    
    # calculate engine speed and sampling ratio in time domain
    eng_spd = df_engspd['spd'].mean()  # rev/s, for one cycle
    eng_rpm = eng_spd * 60             # rpm for this cycle
    sample_rate_ts = sample_rate_rev * eng_rpm / 60  # Hz, in time domain

    # pick sensor data 
    remove_col = ['cycles', 'angles']
    val_col = [col for col in df.columns if col not in remove_col]

    return df.loc[:,val_col].values, sample_rate_ts, eng_rpm


def crank_to_time(df_list, df_list_engspd):
    '''Convert a list of DataFrames from crank angle domain to time domain.

    Parameters:
    -----------
    df_list: list of DataFrame, each contain one sample (engine cycle)

    df_list_engspd: list of DataFrame that contains engine speed.

    Returns:
    --------
    list of numpy array, sampling rate (Hz), and engine speed (rpm)
    '''

    features, sample_rate, eng_spd = [], [], []

    for df, df_engspd in zip(df_list, df_list_engspd):
        # print(type(df), type(df_engspd))
        feat, sr, spd = crank_to_time_single(df, df_engspd)

        features.append(feat)
        sample_rate.append(sr)
        eng_spd.append(spd)

    return features, sample_rate, eng_spd
    

def seq_to_chroma(seq_list, sample_rate_list, method='mfcc'):
    '''Convert sequence features into chroma features in batch.
    Potential to improve efficiency.

    Parameters:
    -----------
    seq_list: list of array, which contain sequential features.

    sample_rate_list: list of decimals, which specify the sampling
    frequency in time domain (Hz).

    method: string, choose between MFCC and MFSC. Some research found MFSC
    works better for deep learning applications.

    Returns:
    --------
    One single array: (batches, map_x1, map_x2, features)
    '''

    chroma_feat = []
    for seq, sr in zip(seq_list, sample_rate_list):
        # librosa's feature processing only accepts 1d data
        coeff = seq_to_chroma_single(seq, sr, method)
        chroma_feat.append(coeff)
    
    chroma_feat = np.stack(chroma_feat, axis=0)

    return chroma_feat


def seq_to_chroma_single(seq_arr, sr, method):
    '''Convert sequence features into chroma features for one DataFrame.
    Note one DataFrame can contain multiple features, so need a for loop.

    Parameters:
    -----------
    seq_arr: 2d array with the shape of (n_seq, n_features)

    sr: sample rate for MFCC calculation.

    method: 'MFCC' or 'MFSC' features.

    Returns:
    --------
    One single array: (map_x1, map_x2, features)
    '''

    coeff = []
    for i in range(seq_arr.shape[-1]):
        if method.lower() == 'mfcc':
            coeff.append(librosa.feature.mfcc(seq_arr[:,i], sr=sr, n_mfcc=20))
        elif method.lower() == 'mfsc':
            # this method need some more tuning
            coeff.append(librosa.feature.melspectrogram(seq_arr[:,i], sr=sr))

    return np.stack(coeff, axis=-1)


def process_knock(data_df, scaler, MA_window=None, resample_rate=None):
    '''
    Preprocessing for knock sensor data, process one cycle only.
    
    Parameters:
    -----------
    data_df: DataFrame of sensor data. First column MUST be angles, the rest are sensor data.

    scaler: sklearn StandardScaler object, must contain same features as data_df columns.

    MA_window: int, moving average window. 

    resample_rate: int, the target sampling rate (per engine cycle). e.g.: set 720 for a resample of 7200 -> 720.

    Returns:
    --------
    sensor_values: numpy array. Scaled, MA'd, resampled features.
    '''

    angles_col = data_df.columns[0:1]
    values_col = data_df.columns[1:]

    # do a moving average, shape remain the same (7200, n)
    if MA_window:
        data_df[values_col] = data_df[values_col].rolling(window=MA_window).mean()
        data_df[values_col] = data_df[values_col].fillna(method='bfill')   # backfill the hole caused by moving avg
    
    # resampling to reduce data
    if resample_rate:
        sensor_values = signal.resample(data_df[values_col], resample_rate)
    else:
        sensor_values = data_df[values_col].values
    
    # scale the data
    if scaler:
        return scaler.transform(sensor_values)
    else:
        return sensor_values


def process_engspd(data_df, scaler, MA_window=None, resample_rate=None):
    '''
    Preprocessing for engine speed data, whose rate is 719 per cycle instead of 7200.
    Pad 1 value to make 720 before using the same process as process_knock().
    '''

    # engine speed data has 719 points per cycle, pad one value in the end.
    # extract column names
    angles_col, spd_col = data_df.columns
    # extract last row values
    angles_val, spd_val = data_df[-1:].values[0]
    angles_val  = angles_val + 1     # 358.5 + 1 = 359.5

    # pad the value
    pad_value = {
        angles_col: angles_val,
        spd_col: spd_val,
    }
    data_df = data_df.append(pad_value, ignore_index=True)
    
    # regular processing
    return process_knock(
        data_df, scaler, 
        MA_window=MA_window,
        resample_rate=resample_rate
        )


def process_peaks(data_df, scaler, resample_rate=120):
    '''
    Process microphone peak (valley) values: find spikes, scale, and resample.
    Because mic signal has spikes (almost) periodically.

    Parameters:
    -----------
    data_df: DataFrame of sensor data. First column MUST be angles, the rest are sensor data.

    scaler: sklearn StandardScaler object, must contain same features as data_df columns.

    resample_rate: int, unit is cycle^-1. Number of sample points per engine cycle.
    This parameter must be specified, as the peaks of acc is irregularly intervaled.
    Default to 120 because for a 7200 data points cycle, the peaks are about 120.

    Returns:
    --------
    peaks_resampled: numpy array. the shape is given by resample_rate.
    '''

    x = data_df['hall'].values

    # find the spikes in acc data
    idx, prop = find_valleys(x, height=2)

    # scale the peak values
    if scaler:
        peaks_scaled = scaler.transform(x[idx].reshape(-1,1))  # scaler defaults to process 2d input
        peaks_scaled = peaks_scaled.squeeze(axis=1)
    else:
        peaks_scaled = x[idx]  # no scaling

    # interpolate to make intervals even, this is NOT optional
    sample_interval = len(x) / resample_rate            # sample interval
    idx_new = np.arange(0, len(x)-1, sample_interval)   # create ts of the resample singal
    peaks_resampled = np.interp(idx_new, idx, peaks_scaled)

    # reshape to 2d
    peaks_resampled = peaks_resampled.reshape(-1,1)  # output 2d, to be consistent

    return peaks_resampled


def find_valleys(x, height=None):
    '''the inverse version of scipy.signal.find_peaks.'''
    x = 0 - x
    peaks, properties = signal.find_peaks(x, 0 - height)
    properties['peak_heights'] = 0 - properties['peak_heights']

    return peaks, properties


def process_mfcc(data_df, sample_rate_ts, scaler=None):
    '''
    Preprocessing for MFCC data, suitable for microphone, accelerometer data.
    
    Parameters:
    -----------
    data_df: DataFrame of sensor data. First column MUST be angles, the rest are sensor data.

    sample_rate_ts: float, voice signal sample rate in TIME DOMAIN. 

    scaler: sklearn StandardScaler object, apply on 2d feature, like an image.


    Returns:
    --------
    mic_mfcc: numpy array. each feature is 2d.
    '''

    angles_col = data_df.columns[0:1]
    values_col = data_df.columns[1:]
    
    mfcc_maps = []

    for col in values_col:
        x = data_df[col].values
        mfcc_maps.append(librosa.feature.mfcc(x, sr=sample_rate_ts, n_mfcc=20))

    # concatenate along a new axis
    mfcc_maps = np.stack(mfcc_maps, axis=-1)

    # apparantly it needs scaling, will need to customize one for 2d data.
    
    return mfcc_maps


def create_dataset(data_tuple, test_split, val_split, batch_size=32, seed=42):
    '''Create a tensorflow dataset: train - val - test.'''
    
    n = len(data_tuple[-1])
    
    val_size = int(n * val_split)
    test_size = int(n * test_split)
    train_size = n - val_size - test_size

    dataset = tf.data.Dataset.from_tensor_slices(data_tuple)
    dataset = dataset.shuffle(buffer_size=n+1, seed=seed)
    train_dataset = dataset.take(train_size).batch(batch_size).prefetch(2)
    val_dataset = dataset.skip(train_size).take(val_size).batch(batch_size).prefetch(2)
    test_dataset = dataset.skip(train_size+val_size).batch(batch_size).prefetch(2)

    return train_dataset, val_dataset, test_dataset


def create_dataset_v2(data_tuple, val_split, test_split, batch_size=32, seed=42):
    '''Create a dataset: train - val - test. using sklearn.
    the original tf version has bug that leaks train set to test set.

    Parameters:
    -----------
    data_tuple: a tuple of (x, y)

    test_split: float number btw (0, 1). Can set 0 to return a None for test dataset

    val_split: float number btw (0, 1). train_split + test_split + val_split = 100%

    Returns:
    --------
    3 tensorflow dataset, one of them could be None depending on the arguments
    '''
    
    x, y = data_tuple
    n = len(y)
    n_classes = len(np.unique(y))  # number of classes

    # determine the size of split
    val_size = int(n * float(val_split))
    test_size = int(n * float(test_split))
    train_size = n - val_size - test_size

    # placeholders for datasets
    train_dataset = None
    val_dataset = None
    test_dataset = None

    # if only two datasets are needed
    if val_size == 0 and test_size == 0:
        raise ValueError('Split value too small. Cannot split dataset with 0 test_split and 0 val_split.')

    # split into train + val
    elif val_size == 0:
        test_size = max(test_size, n_classes)  # make sure test size is at least 3
        test_split = test_size / n

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_split, stratify=y,
            shuffle=True, random_state=seed)
    
    # split into train + test
    elif test_size == 0:
        val_size = max(val_size, n_classes)  # make val test size is at least 3
        

        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=val_split, stratify=y,
            shuffle=True, random_state=seed)
    
    # use the function twice to get 3 datasets
    else:
        val_size = max(val_size, n_classes)  # make val test size is at least 3
        val_split = val_size / n
        test_size = max(test_size, n_classes)  # make sure test size is at least 3
        test_split = test_size / n
        
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_split, stratify=y,
            shuffle=True, random_state=seed)

        val_split = val_split / (1 - test_split)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=val_split, stratify=y_train,
            shuffle=True, random_state=seed)
    
    # make tensorflow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).prefetch(2)
    try: val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    except: pass
    try: test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    except: pass

    print('Date split into train, val, test: {}, {}, {}'.format(train_size, val_size, test_size))
    return train_dataset, val_dataset, test_dataset


def plot_loss_acc(history):
    '''Plot loss and accuracy curve side by side. Input is the training history dict.'''
    
    plt.subplot(1,2,1)
    plt.xlabel('epochs', fontsize=16)
    plt.ylabel('loss', fontsize=16)


    plt.plot(history['loss'], color='blue')
    plt.plot(history['val_loss'], color='red')
    plt.legend(['train_loss','val_loss'], fontsize=16)
    plt.grid()

    plt.subplot(1,2,2)
    plt.xlabel('epochs', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    plt.plot(history['accuracy'], color='blue')
    plt.plot(history['val_accuracy'], color='red')
    plt.legend(['train_acc','val_acc'], fontsize=16)
    plt.ylim([0.4,1.05])
    plt.grid()