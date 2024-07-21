#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.regularizers import l1_l2

import logging
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

import time


# In[2]:


# Tuning parameters
EPOCHS = 150  # The number of round for training.
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
MSE_THRESHOLD = 0.025


# # Source domain and Target domain

# In[3]:


# Import the data set
data_set = pd.read_csv('osp16_train1000.csv')

df = pd.DataFrame(data_set)

X_input_source = df.iloc[:, 0:np.shape(data_set)[1] - 1]  # select feature columns
y_input_source = df.iloc[:, np.shape(data_set)[1] - 1]  # select output column

# Split train set and test set
X_train_source, X_test_source,     y_train_source, y_test_source = train_test_split(X_input_source, 
                                                     y_input_source,
                                                    test_size=0.1,
                                                    random_state=0)

print('The shape of the source X_train is: ', X_train_source.shape)
print(X_train_source)

print('The shape of the source y_train is: ', y_train_source.shape)
print(y_train_source)

# Create the target dataset

X_input_target = df.iloc[1:50000, 0:np.shape(data_set)[1]-1]  # select feature columns
y_input_target = df.iloc[1:50000, np.shape(data_set)[1]-1]  # select output column

# Split train set and test set
X_train_target, X_test_target,     y_train_target, y_test_target = train_test_split(X_input_target, 
                                                     y_input_target, 
                                                     test_size = 0.1, 
                                                     random_state = 1)

print('Shape of the input of the target train dataset is: ', X_train_target.shape)


# # Source model

# In[4]:


# Build a feed-forward DNN
no_unit = 512
input_layer = tf.keras.layers.Input(shape=(X_input_source.shape[1],),name='source_input_layer')
x = tf.keras.layers.Dense(no_unit, activation='relu',
                          name='source_hidden_layer_1')(input_layer)
x = tf.keras.layers.Dense(no_unit, activation='relu', name='source_hidden_layer_2')(x)
x = tf.keras.layers.Dense(no_unit, activation='relu', name='source_hidden_layer_3')(x)
x = tf.keras.layers.Dense(no_unit, activation='relu', name='source_hidden_layer_4')(x)
x = tf.keras.layers.Dense(no_unit, activation='relu', name='source_hidden_layer_5')(x)
output_layer = tf.keras.layers.Dense(1, activation='relu', name='source_output_layer')(x)
source_model = tf.keras.models.Model(input_layer, output_layer)

# generate the DNN model (source model)
optimizer_src = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
source_model.compile(optimizer=optimizer_src,
              loss = tf.keras.losses.MeanSquaredError(),
              metrics=['mse'])

source_model.summary()


# In[5]:


# Improve the training by reducing the learning rate
source_start = time.perf_counter()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=5,
                              min_lr=1e-5)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=5)

history_source = source_model.fit(X_train_source, y_train_source,
                    validation_split=0.1,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[reduce_lr],
                    verbose=1
                    )


# In[6]:


print(history_source.history.keys())  # check metric keys before plotting

src_elapsed = time.perf_counter() - source_start
print('Elapsed %.5f seconds.' % src_elapsed)

plt.figure(figsize=(6, 4))  # set figure ratio
plt.plot(history_source.history['mse'], label='Training')
plt.plot(history_source.history['val_mse'], label='Validation'),
plt.yscale('log')
plt.grid(True)
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.tight_layout()  # avoid missing x-label or y-label
plt.show()


# In[7]:


# the saved training DNN model
source_model.save('trained_source_DNN.keras')

print('--------------------------------------------------------------------\n'
      'evaluation the trained model')

# verify the trained model
source_model_trained = tf.keras.models.load_model('trained_source_DNN.keras')
y_pred_source = source_model_trained.predict(X_test_source)
# y_pred = source_model.predict(X_test)
# Compute the RMSE
RMSE_trained_source = np.sqrt(mean_squared_error(y_test_source, y_pred_source))
print('Source model RMSE is: ', RMSE_trained_source)

if RMSE_trained_source < MSE_THRESHOLD:
    print('Qualified trained source model!')
else:
    print('Re-train the source model.')


# # Target model with frozen and fine-tuned layers
# 
# A common approach is to freeze the initial layers of the model (which often capture more general features) and fine-tune the later layers (which capture more task-specific features). This strategy balances the benefits of both frozen and fine-tuned layers.

# In[8]:


"""
Create the target model
"""

# Freeze some first layers of the source model 
frozen_layers = source_model.layers[:4] # note that layer 4 will not be frozen
for layer in frozen_layers: 
    layer.trainable = False

# Set the some layers of the source model to be trainable (will be fine-tuned in the future)
fine_tuned_layers = source_model.layers[4]
fine_tuned_layers.trainable = True

# Create a new model for fine-tuning
fine_tuned_model = tf.keras.models.Sequential(name='ft_target_model')

# Add the frozen layers to the new model
for layer in frozen_layers:
    fine_tuned_model.add(layer)

# Add the trainable layers (fine-tuned layers) to the new model
fine_tuned_model.add(fine_tuned_layers)

# Add additional layers as needed
fine_tuned_model.add(tf.keras.layers.Dense(no_unit, activation='relu', name='target_added_layer_1'))
fine_tuned_model.add(tf.keras.layers.Dense(no_unit, activation='relu', name='target_added_layer_2'))

# Output layer
fine_tuned_model.add(tf.keras.layers.Dense(1, activation='relu', 
                                           kernel_initializer='normal', 
                                           name='target_output_layer'))

optimizer_ft = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

fine_tuned_model.compile(optimizer = optimizer_ft,
                       loss = 'mse',
                       metrics = ['mse', 'acc'])

fine_tuned_model.summary()


# In[9]:


"""
Train the target model
"""
finetune_start = time.perf_counter()

# train using the target train dataset (TL model training)

history_ft = fine_tuned_model.fit(X_train_target,
                             y_train_target,
                             validation_split = 0.1,
                             epochs = EPOCHS,
                             batch_size = BATCH_SIZE,
                             callbacks = [reduce_lr],
                             verbose = 1
                            )


# In[10]:


print(history_ft.history.keys())  # check metric keys before plotting

ft_elapsed = time.perf_counter() - finetune_start
print('Elapsed %.5f seconds.' % ft_elapsed)

plt.plot(history_ft.history['mse'], label = 'Training')
plt.plot(history_ft.history['val_mse'], label = 'Validation'),
plt.yscale('log')
plt.grid(True)
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.legend(loc = 'upper right')
plt.tight_layout()  # avoid missing x-label or y-label
plt.show()


# In[11]:


fine_tuned_model.save('trained_fine_tuned_model.keras')

"""
Evaluate the trained target model
"""

# verify the trained model
fine_tuned_model_trained = tf.keras.models.load_model('trained_fine_tuned_model.keras')
# y_pred = model_trained.predict(X_test)


# verify the trained model
y_pred_target = fine_tuned_model_trained.predict(X_test_target, verbose=0)

# Compute the RMSE
RMSE_fine_tuned_model = np.sqrt(mean_squared_error(y_test_target.values, y_pred_target))
print('RMSE of the target model is: ', RMSE_fine_tuned_model)


if RMSE_fine_tuned_model < MSE_THRESHOLD:
    print('Qualified trained FT model!')
else:
    print('Re-train the FT model.')


# # Inference on new data

# In[12]:


pos_gpgm = pd.read_csv('data_test1000.csv')
pos_gpgm = pos_gpgm.to_numpy()

snr = pos_gpgm[:, 0]
x_src = pos_gpgm[:, 1]
y_src = pos_gpgm[:, 2]
x_uav = pos_gpgm[:, 3]
y_uav = pos_gpgm[:, 4]
x_des = pos_gpgm[:, 5]
y_des = pos_gpgm[:, 6]
x_eve = pos_gpgm[:, 7]
y_eve = pos_gpgm[:, 8]
z_src = pos_gpgm[:, 9]
z_uav = pos_gpgm[:, 10]
z_des = pos_gpgm[:, 11]
z_eve = pos_gpgm[:, 12]
dSU = pos_gpgm[:, 13]
dUD = pos_gpgm[:, 14]
dUE = pos_gpgm[:, 15]
hSU_ph1 = pos_gpgm[:, 16]
hSU_ph2 = pos_gpgm[:, 17]
hSU_ph3 = pos_gpgm[:, 18]
hSU_ph4 = pos_gpgm[:, 19]
gUD_ph1 = pos_gpgm[:, 20]
gUD_ph2 = pos_gpgm[:, 21]
gUD_ph3 = pos_gpgm[:, 22]
gUD_ph4 = pos_gpgm[:, 23]
gUE_ph1 = pos_gpgm[:, 24]
gUE_ph2 = pos_gpgm[:, 25]
gUE_ph3 = pos_gpgm[:, 26]
gUE_ph4 = pos_gpgm[:, 27]
hSU_re1 = pos_gpgm[:, 28]
hSU_im1 = pos_gpgm[:, 29]
hSU_re2 = pos_gpgm[:, 30]
hSU_im2 = pos_gpgm[:, 31]
hSU_re3 = pos_gpgm[:, 32]
hSU_im3 = pos_gpgm[:, 33]
hSU_re4 = pos_gpgm[:, 34]
hSU_im4 = pos_gpgm[:, 35]
gUD_re1 = pos_gpgm[:, 36]
gUD_im1 = pos_gpgm[:, 37]
gUD_re2 = pos_gpgm[:, 38]
gUD_im2 = pos_gpgm[:, 39]
gUD_re3 = pos_gpgm[:, 40]
gUD_im3 = pos_gpgm[:, 41]
gUD_re4 = pos_gpgm[:, 42]
gUD_im4 = pos_gpgm[:, 43]
gUE_re1 = pos_gpgm[:, 44]
gUE_im1 = pos_gpgm[:, 45]
gUE_re2 = pos_gpgm[:, 46]
gUE_im2 = pos_gpgm[:, 47]
gUE_re3 = pos_gpgm[:, 48]
gUE_im3 = pos_gpgm[:, 49]
gUE_re4 = pos_gpgm[:, 50]
gUE_im4 = pos_gpgm[:, 51]

y_inf_source = np.zeros([pos_gpgm.shape[0],1])
y_inf_finetune = np.zeros([pos_gpgm.shape[0],1])

inference_start = time.perf_counter()

for ii in np.arange(pos_gpgm.shape[0]):
    input_parameters = [snr[ii], x_src[ii], y_src[ii], x_uav[ii], y_uav[ii], x_des[ii],
                        y_des[ii], x_eve[ii], y_eve[ii], z_src[ii], z_uav[ii],
                        z_des[ii], z_eve[ii], dSU[ii], dUD[ii], dUE[ii],
                        hSU_ph1[ii], hSU_ph2[ii], hSU_ph3[ii], hSU_ph4[ii],
                        gUD_ph1[ii], gUD_ph2[ii], gUD_ph3[ii], gUD_ph4[ii],
                        gUE_ph1[ii], gUE_ph2[ii], gUE_ph3[ii], gUE_ph4[ii],
                        hSU_re1[ii], hSU_im1[ii], hSU_re2[ii], hSU_im2[ii],
                        hSU_re3[ii], hSU_im3[ii], hSU_re4[ii], hSU_im4[ii],
                        gUD_re1[ii], gUD_im1[ii], gUD_re2[ii], gUD_im2[ii],
                        gUD_re3[ii], gUD_im3[ii], gUD_re4[ii], gUD_im4[ii],
                        gUE_re1[ii], gUE_im1[ii], gUE_re2[ii], gUE_im2[ii],
                        gUE_re3[ii], gUE_im3[ii], gUE_re4[ii], gUE_im4[ii]]
    x_matlab = np.array(input_parameters)
    x_matlab = x_matlab[np.newaxis,:]
    y_inf_source[ii] = source_model.predict(x_matlab, verbose=0)
    y_inf_finetune[ii] = fine_tuned_model.predict(x_matlab, verbose=0)

np.savetxt("source_pred_24070122_osp16.csv", y_inf_source, delimiter=",")
np.savetxt("finetune_pred_24070122_osp16.csv", y_inf_finetune, delimiter=",")

inf_elapsed = time.perf_counter() - inference_start
print('Elapsed %.5f seconds.' % inf_elapsed)

print('End of the code!')

