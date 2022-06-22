# Thêm các thư viện cần thiết
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras_preprocessing import image
import cv2
import os
import tensorflow as tf
import time
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

#Chỉnh sửa ảnh đầu vào
train_datagen = ImageDataGenerator(rotation_range=0.2,
                                   width_shift_range=0.2, height_shift_range=0.2, vertical_flip=True,
                                   validation_split=0.2, rescale = 1./255, shear_range = 0.2, zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('C:\\Users\\dinht\\Desktop\\Fish_recognize\\data_train',
                                                 target_size = (64, 64),
                                                 batch_size = 12,
                                                 class_mode = 'categorical')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('C:\\Users\\dinht\\Desktop\\Fish_recognize\\data_test',
                                            target_size = (64, 64),
                                            batch_size = 12,
                                            class_mode = 'categorical')

classes = ['ca_ba_duoi','ca_canh_buom_hong','ca_chep_su_tu_trang','ca_hac_dinh_hong','ca_ho_bac','ca_la_han','ca_ma_giap_hoang_kim','ca_mun_panda',
           'ca_neon','ca_phuong_hoang','ca_rong_huyet_long','ca_rong_kim_long','ca_sam_black_diamond','ca_than_tien','ca_tu_van']
print("Image Processing.......Completed")

#Tạo model
model = Sequential()
print("Building Convolution Neural Network.....")
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=15, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

checkpoint = ModelCheckpoint('C://Users//dinht//Desktop//Fish_recognize//model.hdf5', monitor='val_loss', save_best_only = True, mode='auto')
callback_list = [checkpoint]

#Huấn luyện model
print("Training Model")
history=model.fit(x = training_set, validation_data = test_set, batch_size=64, epochs = 20)
step_train = training_set.n
step_val = test_set.n

#Lưu model
model.save('model.h5')
model.save('C://Users//dinht//Desktop//Fish_recognize//model.hdf5')


#Vẽ đồ thị
def plot_history(history_fine):
  acc_f1 = history_fine.history['accuracy']
  val_f1 = history_fine.history['val_accuracy']

  loss = history_fine.history['loss']
  val_loss = history_fine.history['val_loss']

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(acc_f1, label='Acc')
  plt.plot(val_f1, label='Validation Acc')
  plt.legend(loc='lower right')
  plt.title('Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Loss')
  plt.xlabel('epoch')
  plt.show()

def plot_reg_history(history_fine):
  loss = history_fine.history['loss']
  val_loss = history_fine.history['val_loss']
  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Loss')
  plt.xlabel('epoch')
  plt.show()
print("Draw Graphs")
plot_history(history)