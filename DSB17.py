# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

# This is simple script with many limitation due to run on Kaggle CPU server.
# There is used simple CNN with low number of conv layers and filters.
# You can improve this script while run on local GPU just by changing some constants
# It just shows the possible example of dataflow which can be used for solving this problem
import matplotlib.pyplot as plt

conf = dict()
# Change this variable to 0 in case you want to use full dataset
conf['use_sample_only'] = 0
# Save weights
conf['save_weights'] = 0
# How many patients will be in train and validation set during training. Range: (0; 1)
conf['train_valid_fraction'] = 0.7
# Batch size for CNN [Depends on GPU and memory available]
conf['batch_size'] = 1
# Number of epochs for CNN training
conf['nb_epoch'] = 40
conf['nb_epoch'] = 15
# Early stopping. Stop training after epochs without improving on validation
conf['patience'] = 2
# Shape of image for CNN (Larger the better, but you need to increase CNN as well)
conf['image_shape'] = (64, 64)
# Learning rate for CNN. Lower better accuracy, larger runtime.
conf['learning_rate'] = 1e-2
# Number of random samples to use during training per epoch
conf['samples_train_per_epoch'] = 100000
conf['samples_train_per_epoch'] = 500
# conf['samples_train_per_epoch'] = 1
# Number of random samples to use during validation per epoch
conf['samples_valid_per_epoch'] = 3000
conf['samples_valid_per_epoch'] = 100
# conf['samples_valid_per_epoch'] = 1
# Some variables to control CNN structure
conf['level_1_filters'] = 4
conf['level_2_filters'] = 8
conf['dense_layer_size'] = 32
conf['dropout_value'] = 0.2


import dicom
import os
import cv2
import numpy as np
import pandas as pd
import glob
import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.pooling import GlobalMaxPooling1D,GlobalMaxPooling2D,GlobalMaxPooling3D
from keras.layers.convolutional import Convolution1D, MaxPooling1D, ZeroPadding1D,AveragePooling1D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D, AveragePooling3D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import Counter
from my_utils import convert_all_subjects
import time
import math
from keras.layers import BatchNormalization
np.random.seed(2016)
random.seed(2016)


# ____________________________________________________________________________________________________
# Layer (type)                     Output Shape          Param #     Connected to
# ====================================================================================================
# convlstm2d_1 (ConvLSTM2D)        (None, 1, 15, 512, 51 790560      convlstm2d_input_1[0][0]
# ____________________________________________________________________________________________________
# convlstm2d_2 (ConvLSTM2D)        (None, 1, 15, 512, 51 16260       convlstm2d_1[0][0]
# ____________________________________________________________________________________________________
# convolution3d_1 (Convolution3D)  (None, 1, 15, 512, 51 10          convlstm2d_2[0][0]
# ____________________________________________________________________________________________________

def load_and_normalize_dicom(path, x, y, npy=False, z=64, cv3D_size=512):
    if npy:
        dicom_img = np.load(path, mmap_mode='r')
    else:
        dicom1 = dicom.read_file(path)
        dicom_img = dicom1.pixel_array.astype(np.float64)
    mn = dicom_img.min()
    mx = dicom_img.max()
    if (mx - mn) != 0:
        dicom_img = (dicom_img - mn)/(mx - mn)
    else:
        if npy:
            dicom_img[:, :, :] = 0
        else:
            dicom_img[:, :] = 0
    if npy:
        if cv3D_size == 512:
            return dicom_img

        elif dicom_img.shape != (z, x, y):
            # pass
            pad = np.zeros((3, 1))
            pad[0, 0] = max(dicom_img.shape) - dicom_img.shape[0]
            pad[1, 0] = max(dicom_img.shape) - dicom_img.shape[1]
            pad[2, 0] = max(dicom_img.shape) - dicom_img.shape[2]

            paddedInput = np.zeros((max(dicom_img.shape), max(dicom_img.shape), max(dicom_img.shape)))

            paddedInput = np.pad(dicom_img, ((int(math.ceil(pad[0, 0] / 2)),
                                          int(math.floor(pad[0, 0] / 2))), (int(math.ceil(pad[1, 0] / 2)),
                                                                            int(math.floor(pad[1, 0] / 2))),
                                         (int(math.ceil(pad[2, 0] / 2)),
                                          int(math.floor(pad[2, 0] / 2)))), 'constant', constant_values=0)

            paddedInput.resize((256, 256, 256))

            dicom_img = paddedInput
            # dicom_img = cv2.resize(dicom_img, (z, x, y), interpolation=cv2.INTER_CUBIC)
    else:
        if dicom_img.shape != (x, y):
            dicom_img = cv2.resize(dicom_img, (x, y), interpolation=cv2.INTER_CUBIC)
    # print('Type dicom img:')
    # print(type(dicom_img))
    # print(dicom_img.shape)
    return dicom_img

import copy

def batch_generator_train(files, train_csv_table, batch_size, npy=False, cv3D_size=512):
    number_of_batches = np.ceil(len(files)/batch_size)
    counter = 0
    random.shuffle(files)
    next_files = copy.deepcopy(files)
    failed = 0

    while True:
        # batch_files = files[batch_size*counter+failed:batch_size*(counter+1)+failed]
        image_list = []
        mask_list = []
        success = 0
        problem_files=[]

        for f in files[batch_size*counter+failed:]:
            if success==batch_size:
                break
            if npy:
                try:
                    image = load_and_normalize_dicom(f, 64, 64, npy, cv3D_size)
                    success+=1
                except:
                    # print('Problem with:', f)
                    failed+=1
                    try:
                        next_files.remove(f)
                    except:
                        pass
                    problem_files.append(f)
                    batch_files = files[batch_size * counter + failed:batch_size * (counter + 1) + failed]

                    continue
                    # img = np.load(f)
                    # print(img.shape)
                # print(image.shape)
                patient_id = os.path.basename(f.replace('.npy', ''))
            else:
                image = load_and_normalize_dicom(f, conf['image_shape'][0], conf['image_shape'][1])
                patient_id = os.path.basename(os.path.dirname(f))
            is_cancer = train_csv_table.loc[train_csv_table['id'] == patient_id]['cancer'].values[0]
            if is_cancer == 0:
                # mask = [1, 0]
                mask = [1]
            else:
                # mask = [0, 1]
                mask = [0]
            if npy:
                image_list.append(image)
            else:
                image_list.append([image])
            mask_list.append(mask)
        # print(Counter([image[0].shape for image in image_list]))
        counter += 1
        if npy:
            npy_image_list = np.zeros((batch_size, 1, cv3D_size, cv3D_size, cv3D_size))
            for i, image in enumerate(image_list):
                len_image = min(len(image), 512)
                # print(len_image)
                # image = np.expand_dims(np.expand_dims(image[:512], 1),0)
                # print(image.shape)
                # print(len_image)

                npy_image_list[i, 0, :len_image, :, :] = image[:512]

            image_list = npy_image_list
        else:
            image_list = np.array(image_list)
        # print(image_list.shape)
        # print(image_list[0].shape)
        mask_list = np.array(mask_list)
        # print(image_list.shape)
        # print(mask_list.shape)
        if len(mask_list)>0:
            # image_list = np.swapaxes(image_list,1,2)
            image_list.astype(np.int32)
            # print(image_list.shape)

            # print(image_list.nbytes)
            yield image_list, mask_list
        else:
            # print(image_list)
            print(mask_list)
            print(patient_id)
            # print(image_list.shape)
            failed += 1
            try:
                next_files.remove(f)
            except:
                pass

            yield np.zeros((batch_size, 1, cv3D_size, cv3D_size, cv3D_size)), np.array([[0]])

        if counter*batch_size > number_of_batches*batch_size-failed-5:
            print('Reported %s problems with files this epoch' % (len(files)-len(next_files)))
            files = next_files
            next_files = copy.deepcopy(files)
            print('length of new file list: %s' % len(files))

            random.shuffle(files)
            counter = 0
            failed = 0


from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.core import Permute, Reshape

def get_custom_CNN(npy=False, lstm=False, cv3D_size=512):
    model = Sequential()
    if npy:
        if lstm:

            model.add(ConvLSTM2D(nb_filter=2, nb_row=3, nb_col=3, subsample=(2,2),
                               input_shape=(cv3D_size, 1, cv3D_size, cv3D_size), activation='relu',
                               return_sequences=True, dim_ordering='th'))  # border_mode="same",
            print(model.layers[0].input_shape)
            print(model.layers[0].output_shape)
            # model.add(Permute((2,1,3,4)))
            #
            # model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), dim_ordering='th'))
            # model.add(Permute((2,1,3,4)))

            model.add(ConvLSTM2D(nb_filter=4, nb_row=2, nb_col=2, activation='relu', subsample=(2,2),
                               return_sequences=True, dim_ordering='th'))  # border_mode="same"
            print(model.layers[-1].input_shape)
            print(model.layers[-1].output_shape)

            # model.add(ConvLSTM2D(nb_filter=15, nb_row=3, nb_col=3,
            #                    border_mode="same", return_sequences=True))
            model.add(Permute((2,1,3,4)))
            # model.add(MaxPooling3D((1, 508, 508), dim_ordering='th'))
            # model.add(MaxPooling3D((1, 253, 253), dim_ordering='th'))
            model.add(MaxPooling3D((1, 127, 127), dim_ordering='th'))
            print(model.layers[-1].input_shape)
            print(model.layers[-1].output_shape)
            # model.add(Reshape((15,512)))
            model.add(Reshape((4,512)))
            # model.add(Reshape((15,256)))
            model.add(Permute((2,1)))
            model.add(Convolution1D(50,3))
            model.add(BatchNormalization())
            model.add(Dropout(conf['dropout_value']))
            print(model.layers[-1].input_shape)
            print(model.layers[-1].output_shape)

            model.add(GlobalMaxPooling1D())
            print(model.layers[-1].input_shape)
            print(model.layers[-1].output_shape)
            model.add(BatchNormalization())
            model.add(Dropout(conf['dropout_value']))
            model.add(Dense(1, activation='sigmoid'))


            # model.add(Convolution3D(nb_filter=1, kernel_dim1=1, kernel_dim2=3,
            #                       kernel_dim3=3, activation='sigmoid',
            #                       border_mode="same", dim_ordering="th"))
            print(model.layers[-1].input_shape)
            print(model.layers[-1].output_shape)
        else:
            # model.add(
            #     ZeroPadding3D((1, 1, 1), input_shape=(1, 64, 64, 64), dim_ordering='th')),
            model.add(AveragePooling3D((2, 2, 2), strides=(2, 2, 2), dim_ordering='th', input_shape=(1, 512, 512, 512)))
            model.add(Convolution3D(8, 3, 3, 3, activation='relu',  # input_shape=(1, 64, 64, 64),
                                    dim_ordering='th'))
            # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
            model.add(MaxPooling3D((3, 3, 3), strides=(3, 3, 3), dim_ordering='th'))

            model.add(Convolution3D(8, 3, 3, 3, activation='relu', dim_ordering='th'))
            model.add(BatchNormalization())

            model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), dim_ordering='th'))

            # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
            model.add(Convolution3D(8, 2, 2, 2, activation='relu', dim_ordering='th'))
            # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
            model.add(Convolution3D(4, 3, 3, 3, activation='relu', dim_ordering='th'))
            model.add(MaxPooling3D((3, 3, 3), strides=(3, 3, 3), dim_ordering='th'))
    else:
        model.add(ZeroPadding2D((1, 1), input_shape=(1, conf['image_shape'][0], conf['image_shape'][1]), dim_ordering='th'))
        model.add(Convolution2D(conf['level_1_filters'], 3, 3, activation='relu', dim_ordering='th'))
        model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
        model.add(Convolution2D(conf['level_1_filters'], 3, 3, activation='relu', dim_ordering='th'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='th'))

        model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
        model.add(Convolution2D(conf['level_2_filters'], 3, 3, activation='relu', dim_ordering='th'))
        model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
        model.add(Convolution2D(conf['level_2_filters'], 3, 3, activation='relu', dim_ordering='th'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(conf['dense_layer_size'], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(conf['dropout_value']))
    model.add(Dense(conf['dense_layer_size'], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(conf['dropout_value']))

    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=conf['learning_rate'], decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


def get_train_single_fold(train_data, fraction):
    ids = train_data['id'].values
    random.shuffle(ids)
    split_point = int(round(fraction*len(ids)))
    train_list = ids[:split_point]
    valid_list = ids[split_point:]
    return train_list, valid_list

from keras.callbacks import Callback


def create_single_model(npy=False, lstm=False, plot_acc_and_loss=True):

    train_csv_table = pd.read_csv('../input/stage1_labels.csv')
    train_patients, valid_patients = get_train_single_fold(train_csv_table, conf['train_valid_fraction'])
    print('Train patients: {}'.format(len(train_patients)))
    print('Valid patients: {}'.format(len(valid_patients)))


    print('Create and compile model...')
    model = get_custom_CNN(npy, lstm)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=conf['patience'], verbose=0),
        # print_pred()
        # ModelCheckpoint('best.hdf5', monitor='val_loss', save_best_only=True, verbose=0),
    ]
    if npy:
        get_dir = 'stage1_npy'
    else:
        get_dir = 'stage1'
    if conf['use_sample_only'] == 1:
        get_dir = 'sample_images'
    all_patient_size = []
    train_files = []
    for p in train_patients:
        # print(p)
        # print("../input/{}/{}/*.dcm".format(get_dir, p))
        # print(glob.glob("../input/{}/{}/*.dcm".format(get_dir, p)))
        # print'----------------\n'
        if npy:
            new_files = glob.glob("../input/{}/{}.npy".format(get_dir, p))
            # print('-----------')
            # print(glob.glob("../input/{}/{}*.npy".format(get_dir, p)))
            # print(glob.glob("../input/{}/{}/*.dcm".format(get_dir, p)))
            # print(glob.glob("../input/stage1/{}/*.dcm".format(p)))
            if len(new_files)>0:
                train_files += new_files

        else:
            new_files = glob.glob("../input/{}/{}/*.dcm".format(get_dir, p))
            train_files += new_files
        all_patient_size.append(len(new_files))
    print('Number of train files: {}'.format(len(train_files)))

    valid_files = []
    for p in valid_patients:
        if npy:
            new_files = glob.glob("../input/{}/{}.npy".format(get_dir, p))
            # print('-----------')
            # print(glob.glob("../input/{}/{}*.npy".format(get_dir, p)))
            # print(glob.glob("../input/{}/{}/*.dcm".format(get_dir, p)))
            # print(glob.glob("../input/stage1/{}/*.dcm".format(p)))
            if len(new_files)>0:
                valid_files += new_files
        else:
            new_files = glob.glob("../input/{}/{}/*.dcm".format(get_dir, p))

            valid_files += new_files
        all_patient_size.append(len(new_files))
    print('Number of valid files: {}'.format(len(valid_files)))

    print(Counter(all_patient_size))

    class print_pred(Callback):
        def __init__(self):
            self.list_pred = []

        def on_epoch_end(self, epoch, logs={}):
            pred = model.predict_generator(
                batch_generator_train(train_files, train_csv_table, conf['batch_size'], npy=npy), 3, 2)
            print('Prediction for last 3 examples of epoch: %s' %pred)

            self.list_pred.append(pred)

        def on_train_begin(self, logs={}):
            pred = model.predict_generator(
                batch_generator_train(train_files, train_csv_table, conf['batch_size'], npy=npy), 3, 2)
            print('Prediction for first 3 examples: %s' %pred)
            self.list_pred.append(pred)

    callbacks.append(print_pred())

    print('Fit model...')
    print('Samples train: {}, Samples valid: {}'.format(conf['samples_train_per_epoch'], conf['samples_valid_per_epoch']))
    fit = model.fit_generator(generator=batch_generator_train(train_files, train_csv_table, conf['batch_size'], npy=npy),
                          nb_epoch=conf['nb_epoch'],
                          samples_per_epoch=conf['samples_train_per_epoch'],
                          validation_data=batch_generator_train(valid_files, train_csv_table, conf['batch_size'], npy=npy),
                          nb_val_samples=conf['samples_valid_per_epoch'],
                          verbose=1,
                          callbacks=callbacks, max_q_size=5, nb_worker=2)

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

    return model


def create_submission(model, npy=False):
    sample_subm = pd.read_csv("../input/stage1_sample_submission.csv")
    ids = sample_subm['id'].values
    all_patient_size = []
    for index_id, id in enumerate(ids):
        print('Predict for patient {}, {} of {}'.format(id, index_id+1, len(ids)))
        if npy:
            files = glob.glob("../input/stage1_npy/{}.npy".format(id))
        else:
            files = glob.glob("../input/stage1/{}/*.dcm".format(id))
        image_list = []
        all_patient_size.append(len(files))
        for f in files:
            image = load_and_normalize_dicom(f, conf['image_shape'][0], conf['image_shape'][1],npy)
            if npy:
                image_list.append(image)
            else:
                image_list.append([image])
        if npy:
            npy_image_list = np.zeros((len(image_list), 1, 512, 512, 512))
            for i, image in enumerate(image_list):
                len_image = min(len(image), 512)

                npy_image_list[i, 0, :len_image, :, :] = image[:512]
                # print(len_image)
                # image = np.expand_dims(np.expand_dims(image[:512], 1),0)
                # print(image.shape)
                # print(len_image)

                # npy_image_list[i, 0, :len_image, :, :] = image[:512]

            image_list = npy_image_list
        else:
            image_list = np.array(image_list)
        batch_size = len(image_list)
        if batch_size>0:
            image_list = np.swapaxes(image_list,1,2)

            predictions = model.predict(image_list, verbose=1, batch_size=batch_size)
            pred_value = predictions[:].mean()
            if index_id%1==0:
                print('pred %s' %pred_value)
        else:
            pred_value=0.5
        sample_subm.loc[sample_subm['id'] == id, 'cancer'] = pred_value

    print(Counter(all_patient_size))
    sample_subm.to_csv("subm_3d_cnn.csv", index=False)


import sys


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile_fulltrain.txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

sys.stdout = Logger()

# convert_all_subjects()

t0 = time.time()
npy=True
lstm=False
model = create_single_model(npy, lstm)
model.save_weights('3dcnn.h5')
create_submission(model,npy)

print('Time taken on model: %s' %(time.time()-t0))

t1 = time.time()


print('Time taken on converting: %s' %(time.time()-t1))

print('Total time taken: %s' %(time.time()-t0))

sys.stdout.close()