import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf

# downloading data set (audio files)
# !wget https://zenodo.org/record/1342401/files/Jakobovski/free-spoken-digit-dataset-v1.0.8.zip?download=1

# extracting audio files
# filename = "/content/free-spoken-digit-dataset-v1.0.8.zip?download=1"
# from zipfile import ZipFile
# archive = ZipFile(filename, 'r')
# archive.extractall()

n_fft = 512 #windows size
hop_length = 256 #window slide
n_mels = 40
f_min = 20
f_max = 3000 #maximum human can produce
sample_rate = f_max * 2

def one_hot_encoding(data):
  temp = [0] * 10
  temp[data] = 1
  return temp
  
directory_path = '/content/Jakobovski-free-spoken-digit-dataset-e9e1155/recordings/'
audio_files = os.listdir(directory_path)
audio_data_set = []
labels = []
for audio_file in audio_files:
  data, data_rate= librosa.load(directory_path + audio_file, sr=sample_rate, res_type= 'kaiser_fast')
  audio_data_set.append(data)
  label = one_hot_encoding(int(audio_file[0]))
  labels.append(label)
  
path_to_save = '/content/spectogram/'
spec_data_set = []
for data in audio_data_set:
  mel_spec = librosa.feature.melspectrogram(data, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, sr=sample_rate, power=1.0, fmin=f_min, fmax=f_max)
  mel_spec_db = librosa.amplitude_to_db(np.abs(mel_spec), ref=np.max)
  mel_spec_db = cv2.resize(mel_spec_db,(40,40))
  mel_spec_db.shape = (40,40,1)
  spec_data_set.append(mel_spec_db)
  
input_img = tf.placeholder(dtype=tf.float64, shape = (None,40,40,1))
label = tf.placeholder(dtype=tf.float64, shape = (None,10))
print(input_img.shape)
conv_1 = tf.layers.conv2d(inputs= input_img, filters= 256, kernel_size= 5, strides= 2, padding='same')
conv_1 = tf.maximum(conv_1, conv_1 * 0.2)
print(conv_1.shape)
# 20x20x256

conv_2 = tf.layers.conv2d(inputs= conv_1, filters= 128, kernel_size= 5, strides= 2, padding='same')
conv_2 = tf.maximum(conv_2, conv_2 * 0.2)
# 10x10x128

conv_3 = tf.layers.conv2d(inputs= conv_2, filters= 64, kernel_size= 5, strides= 2, padding='same')
conv_3 = tf.maximum(conv_3, conv_3 * 0.2)
# 5x5x64

conv_3_reshape = tf.reshape(conv_3,  shape= (-1, 5*5*64))

dense_1 = tf.layers.dense(inputs= conv_3_reshape, units= 64)
dense_1 = tf.maximum(dense_1, dense_1 * 0.2)

logits = tf.layers.dense(inputs= dense_1, units= 10)

prob = tf.nn.softmax(logits)

error = tf.nn.softmax_cross_entropy_with_logits(labels=label ,logits= logits)
train = tf.train.AdamOptimizer().minimize(error)

spec_data_set = np.asarray(spec_data_set)
labels = np.asarray(labels)
with tf.device('/GPU:0'):
  batch_size = 50
  epochs = 100
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  batches_x = [spec_data_set[i:i+batch_size] for i in range(0,spec_data_set.shape[0],batch_size)]
  batches_y = [labels[i:i+batch_size] for i in range(0,labels.shape[0],batch_size)]
  for e in range(epochs):
    for batch_x,batch_y in zip(batches_x,batches_y):
      sess.run(train , feed_dict= {input_img:batch_x, label:batch_y})
    if e%(epochs/10) == 0:
      t_error = sess.run(error , feed_dict= {input_img:spec_data_set, label:labels})
      print('e:',e,'total_error:',np.mean(t_error))

i = 100 #ith dataset
print(np.argmax(sess.run(prob , feed_dict= {input_img:[spec_data_set[i]]}) == labels[i])
