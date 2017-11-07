import numpy as np
import scipy.fftpack as sfp
import pyaudio
import pandas as pd
from sklearn import svm
import time

# Instantiations
audio = pyaudio.PyAudio()

# Load the CSV files
dataset = pd.read_csv('voice_mod.csv')
hyparams = pd.read_csv('hyparams_svm_results_sklearn.csv')

# Split the Dataset into Features and Target
x_train = np.asarray([[row['IQR'], row['meanfun']] for index, row in dataset.iterrows()])
y_train = np.asarray([[row['label']] for index, row in dataset.iterrows()]).ravel()

# Obtain the Hyper Parameters from File
hps = hyparams.as_matrix()
lin_c = hps[0][1]
rbf_c, rbf_sigma = hps[1][1], hps[1][3]
sigmoid_c, sigmoid_gamma, sigmoid_coef = hps[2][1], hps[2][3], hps[2][4]
poly_c, poly_deg, poly_gamma, poly_coef = hps[3][1], hps[3][2], hps[3][3], hps[3][4]

# This method is used if there is more than one Microphone. Also includes Wireless Devices.
def get_All_Mics():
    all_mics_index = []
    for i in range(audio.get_device_count()):
        devices = audio.get_device_info_by_index(i)
        if (devices['maxOutputChannels'] == 0):
            if (('Microphone' in devices['name']) or ('Head' in devices['name'])):
                all_mics_index.append(devices['index'])
    return all_mics_index

def record():
    CHUNK = 1000
    FORMAT = pyaudio.paFloat32
    CHANNELS = 2
    RATE = 48000
    RECORD_SECONDS = 3
    
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=get_All_Mics()[0], 
                        output=False, frames_per_buffer=CHUNK)
    print("\nRecording Voice ...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("\nRecord Complete !")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    return frames

def read_in_data(in_data):
    in_data = np.fromstring(in_data, dtype=np.float32)
    return in_data

def getAudioData(frames):
    audio = b''.join(frames)
    audio = read_in_data(audio)
    audio[audio <= 0.0] = 0.0
    audio = audio[audio != 0.0]
    return audio

def getFreqs(audio):
    freq_data = sfp.rfft(audio)
    freq_data = np.abs(freq_data)
    freq_data = freq_data[freq_data <= 280.0]
    freq_data = freq_data[freq_data != 0.0]
    return freq_data

def getFundFreqs(audio):
    # Calculate Mean Frequency
    freq_data = sfp.rfft(audio)
    freq_data = np.abs(freq_data)
    freq_data = freq_data[freq_data <= 280.0]
    freq_data = freq_data[freq_data != 0.0]
    freq_mean = np.abs(np.mean(freq_data))
    return freq_mean

def getQuartiles(audio):
    # Calculate Interquartile Range
    freq_data = sfp.rfft(audio)
    freq_data = np.abs(freq_data)
    freq_data = freq_data[freq_data <= 280.0]
    freq_data = freq_data[freq_data != 0.0]
    q25 = np.median(freq_data[ : int(len(freq_data)/2)])
    q75 = np.median(freq_data[int(len(freq_data)/2) : ])
    iqr = np.abs(q75 - q25)
    return iqr

def predictGenderStat(params):
    meanfun, iqr = params[0], params[1]
    if meanfun < 0.14:
        if iqr >= 0.07:
            gender = 'MALE'
        else:
            gender = 'FEMALE'
    else:
        gender = 'FEMALE'
    return gender

def fit(params):
    x_train, y_train = params[0], params[1]
    k, a, d, g, c = params[2], params[3], params[4], params[5], params[6]
    cache = params[7]
    svc = svm.SVC(kernel=k, C=a, degree=d, gamma=g, coef0=c, cache_size=cache)
    t = time.time()
    model_fit = svc.fit(x_train, y_train)
    svc_fit = time.time() - t
    print('Time Taken to Fit the Model : ' + str(svc_fit) + ' Secs')
    return model_fit

def predict(svc_fit, iqr, meanfun):
    t = time.time()
    y_pred = svc_fit.predict(np.array([iqr, meanfun]).reshape(1, -1))
    svc_predict = time.time() - t
    print('Time Taken to Predict using the Fitted Model : ' + str(svc_predict) + ' Secs')
    if y_pred == 0:
        y_pred = 'FEMALE'
    else:
        y_pred = 'MALE'
    return y_pred

# Record Audio Sample and Analyze
frames = record()
audio = getAudioData(frames)
freqs = getFreqs(audio)
meanfun = getFundFreqs(audio)
iqr = getQuartiles(audio)
print('\nIQR : %f and FundFreqMean : %f \n' % (iqr, meanfun))
gender = predictGenderStat([meanfun, iqr])
print ('Arithmetic Statistical Model Predicted You Are A : ' + str(gender) + ' ! \n')

# Predictions
# Linear
lin_fit = fit([x_train, y_train, 'linear', lin_c, 1.0, np.sqrt(0.5), 0.0, 4096])
lin_pred = predict(lin_fit, iqr, meanfun)
print ('Linear SVM Model Predicted You Are A : ' + str(lin_pred) + ' ! \n')
# Polyynomial
poly_fit = fit([x_train, y_train, 'poly', poly_c, poly_deg, poly_gamma, poly_coef, 4096])
poly_pred = predict(poly_fit, iqr, meanfun)
print ('Polynomial SVM Model Predicted You Are A : ' + str(poly_pred) + ' ! \n')
# Gaussian
rbf_fit = fit([x_train, y_train, 'rbf', rbf_c, 1.0, rbf_sigma, 0.0, 4096])
rbf_pred = predict(rbf_fit, iqr, meanfun)
print ('Gaussian SVM Model Predicted You Are A : ' + str(rbf_pred) + ' ! \n')
# Sigmoid
sigmoid_fit = fit([x_train, y_train, 'sigmoid', sigmoid_c, 1.0, sigmoid_gamma, sigmoid_coef, 4096])
sigmoid_pred = predict(sigmoid_fit, iqr, meanfun)
print ('Sigmoidal SVM Model Predicted You Are A : ' + str(sigmoid_pred) + ' ! \n')


# End of File