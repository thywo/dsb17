from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pickle
import h5py
import cv2
import os
import glob
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Lambda, Activation
from keras.models import Model

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
from my_utils_feat import batch_generator_dat

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



######################################################
######################################################


inp = Input(600, 512, 32, 32)
x = TimeDistributed(Masking(mask_value=0.0))(inp)
x = TimeDistributed(lrg_layers(x))
x = BatchNormalization()(x)
x = Dense(2,activation='softmax')(x)
model = Model(inp, x)
model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

train_files =

batch_size = 1
model.fit_generator(batch_generator_dat(train_files,train_csv_table,batch_size), trn_labels, batch_size=batch_size, nb_epoch=3,
             validation_data=([conv_val_feat, val_sizes], val_labels))





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




















