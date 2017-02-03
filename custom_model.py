from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pickle
import h5py
import cv2
import os
import glob
import random
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Lambda, Activation
from keras.models import Model
from keras.callbacks import Callback
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Convolution1D, MaxPooling1D
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D, AveragePooling3D
from keras.layers.pooling import GlobalMaxPooling3D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Input, Dense
from keras.layers.core import Masking
from keras.optimizers import Adadelta, Adam
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, History
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

from collections import Counter
import sys
import shutil
from keras.layers.wrappers import TimeDistributed
from my_utils_feat import batch_generator_dat, load_array

def lrg_layers(x, nf=128, p=0.):

    x = BatchNormalization(axis=1)(x)
    x = Convolution2D(nf,3,3, activation='relu', border_mode='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D()(x)

    x = Convolution2D(nf,3,3, activation='relu', border_mode='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D()(x)

    x = Convolution2D(nf,3,3, activation='relu', border_mode='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D((1,2))(x)

    x = Convolution2D(8,3,3, border_mode='same')(x)
    x = Dropout(p)(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    return x


class print_pred(Callback):
    def __init__(self):
        self.list_pred = []

    def on_epoch_end(self, epoch, logs={}):
        pred = model.predict_generator(
           batch_generator_dat(train_files,train_csv_table,batch_size, pad=pad_length),3,2)
        print('Prediction for last 3 examples of epoch: %s' %pred)

        self.list_pred.append(pred)

    def on_train_begin(self, logs={}):
        pred = model.predict_generator(
            batch_generator_dat(train_files,train_csv_table,batch_size, pad=pad_length), 3, 2)
        print('Prediction for first 3 examples: %s' %pred)
        self.list_pred.append(pred)


def get_train_single_fold(train_data, fraction):
    ids = train_data['id'].values
    random.shuffle(ids)
    split_point = int(round(fraction*len(ids)))
    train_list = ids[:split_point]
    valid_list = ids[split_point:]
    return train_list, valid_list


def create_submission(model, submission_name="subm_vgg_feat.csv", pad=600):
    sample_subm = pd.read_csv("../input/stage1_sample_submission.csv")
    ids = sample_subm['id'].values
    all_patient_size = []
    for index_id, id in enumerate(ids):
        print('Predict for patient {}, {} of {}'.format(id, index_id+1, len(ids)))
        # files = glob.glob("../input/stage1/{}/*.dcm".format(id))
        files = glob.glob("../input/results/conv_ft_512_{}.dat".format(id))

        image_list = []
        all_patient_size.append(len(files))
        for f in files:
            image = load_array(f)
            padded_image = np.zeros((600, 512, 32, 32))
            len_image = min(len(image), pad)
            # print(len_image)
            # image = np.expand_dims(np.expand_dims(image[:512], 1),0)
            # print(image.shape)
            # print(len_image)

            padded_image[:len_image, :, :, :] = image[:pad]
            image_list.append([padded_image])

        batch_size = len(image_list)
        if batch_size>0:
            # image_list = np.swapaxes(image_list,1,2)

            predictions = model.predict(image_list, verbose=1, batch_size=batch_size)
            pred_value = predictions[:, 1].mean()
            pred_value_2 = predictions[0, 1].mean()
            if index_id%1==0:
                print('pred %s' %pred_value)
                print('pred_2 %s' %pred_value_2)
        else:
            pred_value=0.5
        sample_subm.loc[sample_subm['id'] == id, 'cancer'] = pred_value

    print(Counter(all_patient_size))
    sample_subm.to_csv(submission_name, index=False)
######################################################
######################################################
pad_length=600

inp = Input(pad_length, 512, 32, 32)
x = TimeDistributed(Masking(mask_value=0.0))(inp)
x = TimeDistributed(lrg_layers(x))
x = BatchNormalization()(x)
x = Dense(2,activation='softmax')(x)
model = Model(inp, x)
model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

train_csv_table = pd.read_csv('../input/stage1_labels.csv')
train_patients, valid_patients = get_train_single_fold(train_csv_table, 0.2)
print('Train patients: {}'.format(len(train_patients)))
print('Valid patients: {}'.format(len(valid_patients)))
all_patient_size = []
train_files = []
for p in train_patients:
    new_files = glob.glob("../input/results/conv_ft_512_{}.dat".format(p))
    train_files += new_files
    all_patient_size.append(len(new_files))
print('Number of train files: {}'.format(len(train_files)))

valid_files = []
for p in valid_patients:
    new_files = glob.glob("../input/results/conv_ft_512_{}.dat".format(p))

    valid_files += new_files
    all_patient_size.append(len(new_files))
print('Number of valid files: {}'.format(len(valid_files)))

print(Counter(all_patient_size))

plot_acc_and_loss = True

patience = 3
n_epoch = 10
batch_size = 1
callbacks = list(
EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
# ModelCheckpoint('best.hdf5', monitor='val_loss', save_best_only=True, verbose=0),
)

callbacks.append(print_pred())

fit = model.fit_generator(batch_generator_dat(train_files,train_csv_table,batch_size, pad=pad_length), len(train_files),
                    n_epoch,callbacks=callbacks,
                    validation_data=batch_generator_dat(valid_files,train_csv_table,batch_size, pad=pad_length),
                    nb_val_samples=len(valid_files),class_weight={},max_q_size=2)


if plot_acc_and_loss:
    # summarize history for accuracy
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fit.history['acc']);
    plt.plot(fit.history['val_acc']);
    plt.title('model accuracy');
    plt.ylabel('accuracy');
    plt.xlabel('epoch');
    plt.legend(['train', 'valid'], loc='upper left');

    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(fit.history['loss']);
    plt.plot(fit.history['val_loss']);
    plt.title('model loss');
    plt.ylabel('loss');
    plt.xlabel('epoch');
    plt.legend(['train', 'valid'], loc='upper left');
    plt.show()

create_submission(model,"vgg_feat_v1.csv", pad=pad_length)
#
# x1 = BatchNormalization(axis=1)(inp)
# x1 = incep_block(x1)
# x1 = incep_block(x1)
# x1 = incep_block(x1)
# x1 = Dropout(0.75)(x1)
# x1_class = Convolution2D(8,3,3, border_mode='same')(x1)
# x1_bb = Convolution2D(4,3,3, border_mode='same')(x1)
# x1_class = GlobalAveragePooling2D()(x1_class)
# # x1_class = GlobalMaxPooling2D()(x1_class)
# outp1_bb = GlobalAveragePooling2D()(x1_bb)
# outp1_class = Activation('softmax')(x1_class)
# # outp1_bb = x1_bb
#
# p=0.6
# x2 = MaxPooling2D()(inp)
# x2 = BatchNormalization(axis=1)(x2)
# x2 = Dropout(p/4)(x2)
# x2 = Flatten()(x2)
# x2 = Dense(512, activation='relu')(x2)
# x2 = BatchNormalization()(x2)
# x2 = Dropout(p)(x2)
# x2 = Dense(512, activation='relu')(x2)
# x2 = BatchNormalization()(x2)
# x2 = Dropout(p/2)(x2)
# outp2_bb = Dense(4, name='bb')(x2)
# # outp2_bb = two_dense_block(x2, output_dim=2)
# # outp2_bb = two_dense_block(x2, output_dim=4, mode='mul')
# outp2_class = Dense(8, activation='softmax', name='class')(x2)
#
# # outp1_bb = copy.deepcopy(outp1_bb)
# # outp2_bb = copy.deepcopy(outp2_bb)
# # outp1_class = copy.deepcopy(outp1_class)
# # outp2_class = copy.deepcopy(outp2_class)
#
# outp_class = merge([outp1_class, outp2_class], mode='ave')
# outp_bb = merge([outp1_bb, outp2_bb], mode='ave')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#




















