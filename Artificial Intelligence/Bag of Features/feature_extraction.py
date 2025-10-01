import os
import numpy as np
import librosa
from functions import *
def extract_features(audio_set):
  hop_size = 11.6
  FFT_size = 1024
  freq_min = 0
  mel_filter_num = 36
  dct_filter_num = 21

  s = 2
  y_test = []
  diccionario = {'dream':0, 'energetic':1, 'happy':2, 'sad':3}
  directoryFirst = audio_set
  elementos_directorio = os.listdir(directoryFirst)
  data = np.zeros((2584,20))
  for a, folder_place in enumerate(elementos_directorio):
    directorio = os.path.join(directoryFirst, folder_place)
    datos = os.listdir(directorio)
    for m, files in enumerate(datos):
      directorioMusic = os.path.join(directorio,files)
      #Load the audio
      audio, sample_rate = librosa.load(directorioMusic)
      freq_high = sample_rate / 2
      #Normalize the audio
      audio = normalize_audio(audio)
      #Slide the audio
      audio_framed = frame_audio(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
      window = get_window("hann", FFT_size, fftbins=True)
      audio_win = audio_framed * window
      audio_winT = np.transpose(audio_win)
      audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
      
      #MFCCs
      #Fast Fourier Transform
      for n in range(audio_fft.shape[1]):
          audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
      audio_fft = np.transpose(audio_fft)
      
      #Create Filters 
      filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=sample_rate)
      filters = get_filters(filter_points, FFT_size)
      enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
      filters *= enorm[:, np.newaxis]
      
      #Power Spectrum
      audio_power = np.square(np.abs(audio_fft))
      
      #Apply Filters
      audio_filtered = np.dot(filters, np.transpose(audio_power))
      audio_filtered[audio_filtered == 0] = np.nextafter(0, 1)
      
      #Convert to Log
      audio_log = 10.0 * np.log10(audio_filtered)
      
      #Discrete Cosine Transform
      dct_filters = dct(dct_filter_num, mel_filter_num)
      cepstral_coefficents = np.dot(dct_filters, audio_log)
      cepstral_coefficents = np.transpose(cepstral_coefficents[1:, :])
      data = np.vstack((data,cepstral_coefficents[:1292,:]))
      #Save the label
      y_test.append(diccionario[files.split('_')[0]])

      #Tempogram
      hop_length = 512
      oenv = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)
      ac_tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sample_rate, hop_length=hop_length, norm=None)
      ac_tempogramT = abs(np.transpose(ac_tempogram))
      data = np.vstack((data,ac_tempogramT[:1292,:20]))
  y_train = data[2584:,:]
  y_test = np.array(y_test)
  return y_train, y_test
