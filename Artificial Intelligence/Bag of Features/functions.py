import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import IPython.display as ipd
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq
from sklearn.metrics.pairwise import chi2_kernel
import librosa

sample_rate = 44100
def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

def frame_audio(audio, FFT_size=1024, hop_size=11.6, sample_rate=sample_rate): #hop_size in seconds
    audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int) #Getting samples per hope
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num, FFT_size))
    for n in range(frame_num):
        frames[n] = audio[n*frame_len:n*frame_len+FFT_size]
    return frames

def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def mel_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=sample_rate):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = mel_to_freq(mels)
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])

    return filters
def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)
    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
    return basis
    
def array_normalization(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = np.interp(array, (min_val, max_val), (0.001, 1.0))
    return normalized_array
    
def histogram(y,X, centroides):
  data = []
  av = 0
  for i in range(len(y)):
    im_featuresTr = np.zeros(centroides.shape[0])
    for j in range(2584):
      words,_ = vq(X[av][np.newaxis,:],centroides)#abs(np.sum((X[av] / centroides) - np.log(centroides / X[av])-1,axis=1)).argmin() #
      im_featuresTr[words[0]] += 1
      av += 1
    data.append(im_featuresTr)
  return np.array(data)
    

