from __future__ import division,print_function
import math, os, json, sys, re
import cPickle as pickle
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter, attrgetter, methodcaller
from collections import OrderedDict
import itertools
from itertools import chain

import pandas as pd
import PIL
from PIL import Image
from numpy.random import random, permutation, randn, normal, uniform, choice
from numpy import newaxis
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from sklearn.metrics import confusion_matrix
import bcolz
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

from IPython.lib.display import FileLink

import theano
from theano import shared, tensor as T
from theano.tensor.nnet import conv2d, nnet
from theano.tensor.signal import pool

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.layer_utils import layer_from_config
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer

from vgg16 import *
from vgg16bn import *
np.set_printoptions(precision=4, linewidth=100)


to_bw = np.array([0.299, 0.587, 0.114])
def gray(img):
    return np.rollaxis(img,0,3).dot(to_bw)
def to_plot(img):
    return np.rollaxis(img, 0, 3).astype(np.uint8)
def plot(img):
    plt.imshow(to_plot(img))


def floor(x):
    return int(math.floor(x))
def ceil(x):
    return int(math.ceil(x))

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        if titles is not None:
            sp.set_title(titles[i], fontsize=18)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


def do_clip(arr, mx):
    clipped = np.clip(arr, (1-mx)/1, mx)
    return clipped/clipped.sum(axis=1)[:, np.newaxis]


def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical',
                target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


def onehot(x):
    return to_categorical(x)


def wrap_config(layer):
    return {'class_name': layer.__class__.__name__, 'config': layer.get_config()}


def copy_layer(layer): return layer_from_config(wrap_config(layer))


def copy_layers(layers): return [copy_layer(layer) for layer in layers]


def copy_weights(from_layers, to_layers):
    for from_layer,to_layer in zip(from_layers, to_layers):
        to_layer.set_weights(from_layer.get_weights())


def copy_model(m):
    res = Sequential(copy_layers(m.layers))
    copy_weights(m.layers, res.layers)
    return res


def insert_layer(model, new_layer, index):
    res = Sequential()
    for i,layer in enumerate(model.layers):
        if i==index: res.add(new_layer)
        copied = layer_from_config(wrap_config(layer))
        res.add(copied)
        copied.set_weights(layer.get_weights())
    return res


def adjust_dropout(weights, prev_p, new_p):
    scal = (1-prev_p)/(1-new_p)
    return [o*scal for o in weights]


def get_data(path, target_size=(224,224)):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range(batches.nb_sample)])


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname, dim):
    array = bcolz.open(fname, mode='a')
    print(sys.getsizeof(array))
    print(type(array))
    print(array.chunklen)
    print(array.cbytes)
    print(array.dtype)
    print(array.itemsize)
    print(array.len)
    print(array.nbytes)
    print(array.shape)
    array.resize(dim)
    print(array.shape)
    print(array.nchunks)
    print(array.nleftover)
    array.free_cachemem()
    print(array.nbytes)
    # try:
    #     array=array[:]
    # except:
    #     print("couldn't load numpy")
    #     array = np.zeros((dim,512,32,32))
    print(array.shape)
    print(sys.getsizeof(array))
    return array


def mk_size(img, r2c):
    r,c,_ = img.shape
    curr_r2c = r/c
    new_r, new_c = r,c
    if r2c>curr_r2c:
        new_r = floor(c*r2c)
    else:
        new_c = floor(r/r2c)
    arr = np.zeros((new_r, new_c, 3), dtype=np.float32)
    r2=(new_r-r)//2
    c2=(new_c-c)//2
    arr[floor(r2):floor(r2)+r,floor(c2):floor(c2)+c] = img
    return arr


def mk_square(img):
    x,y,_ = img.shape
    maxs = max(img.shape[:2])
    y2=(maxs-y)//2
    x2=(maxs-x)//2
    arr = np.zeros((maxs,maxs,3), dtype=np.float32)
    arr[floor(x2):floor(x2)+x,floor(y2):floor(y2)+y] = img
    return arr


def vgg_ft(out_dim):
    vgg = Vgg16()
    vgg.ft(out_dim)
    model = vgg.model
    return model

def vgg_ft_bn(out_dim):
    vgg = Vgg16BN()
    vgg.ft(out_dim)
    model = vgg.model
    return model


def get_classes(path):
    batches = get_batches(path+'train', shuffle=False, batch_size=1)
    val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
    # test_batches = get_batches(path+'test', shuffle=False, batch_size=1)
    test_batches = val_batches
    return (val_batches.classes, batches.classes, onehot(val_batches.classes), onehot(batches.classes),
        val_batches.filenames, batches.filenames, test_batches.filenames)


def split_at(model, layer_type):
    layers = model.layers
    layer_idx = [index for index,layer in enumerate(layers)
                 if type(layer) is layer_type][-1]
    return layers[:layer_idx+1], layers[layer_idx+1:]


class MixIterator(object):
    def __init__(self, iters):
        self.iters = iters
        self.multi = type(iters) is list
        if self.multi:
            self.N = sum([it[0].N for it in self.iters])
        else:
            self.N = sum([it.N for it in self.iters])

    def reset(self):
        for it in self.iters: it.reset()

    def __iter__(self):
        return self

    def next(self, *args, **kwargs):
        if self.multi:
            nexts = [[next(it) for it in o] for o in self.iters]
            n0s = np.concatenate([n[0] for n in o])
            n1s = np.concatenate([n[1] for n in o])
            return (n0, n1)
        else:
            nexts = [next(it) for it in self.iters]
            n0 = np.concatenate([n[0] for n in nexts])
            n1 = np.concatenate([n[1] for n in nexts])
            return (n0, n1)


import dicom
import os
import random

import cv2
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
            print('resizing from %s' % dicom_img.shape)
            dicom_img = cv2.resize(dicom_img, (x, y), interpolation=cv2.INTER_CUBIC)
    # print('Type dicom img:')
    # print(type(dicom_img))
    # print(dicom_img.shape)
    return dicom_img

import copy
def batch_generator_train(files, train_csv_table, batch_size, npy=False, cv3D_size=512, from_gray_to_rgb=False):
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

        # for f in files[batch_size*counter+failed:]:
        for f in files[batch_size * counter:batch_size * (counter+1)]:
            # if success==batch_size:
            #     break
            if npy:
                try:
                    image = load_and_normalize_dicom(f, cv3D_size, cv3D_size, npy, cv3D_size)
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
                image = load_and_normalize_dicom(f, cv3D_size, cv3D_size)

                patient_id = os.path.basename(os.path.dirname(f))
                success += 1
            try:
                is_cancer = train_csv_table.loc[train_csv_table['id'] == patient_id]['cancer'].values[0]

                if is_cancer == 0:
                    mask = [1, 0]
                    # mask = [1]
                else:
                    mask = [0, 1]
                    # mask = [0]
            except:
                # print('Problem with %s' % patient_id)
                mask = [0.5, 0.5]
            if npy:
                if from_gray_to_rgb:
                    image_list.append(np.repeat(image,3, axis=0))
                else:
                    image_list.append(image)
            else:
                if from_gray_to_rgb:
                    image_list.append([image]*3)
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

        if counter== number_of_batches:
            # print('Reported %s problems with files this epoch' % (len(files)-len(next_files)))
            # files = next_files
            # next_files = copy.deepcopy(files)
            # print('length of new file list: %s' % len(files))

            random.shuffle(files)
            counter = 0
            failed = 0


# if counter * batch_size > number_of_batches * batch_size - failed - 5:
#     print('Reported %s problems with files this epoch' % (len(files) - len(next_files)))
#     files = next_files
#     next_files = copy.deepcopy(files)
#     print('length of new file list: %s' % len(files))
#
#     random.shuffle(files)
#     counter = 0
#     failed = 0


def batch_generator_dat(files, train_csv_table, batch_size, pad=600, print_padded=False, number=1):
    print('new generator created')
    number_of_batches = np.ceil(len(files)/batch_size)
    counter = 0
    random.shuffle(files)
    # next_files = copy.deepcopy(files)
    # failed = 0

    while True:
        # batch_files = files[batch_size*counter+failed:batch_size*(counter+1)+failed]
        image_list = []
        mask_list = []
        success = 0
        problem_files=[]
        batch_files = files[batch_size * counter:batch_size * (counter+1)]
        # for f in files[batch_size*counter+failed:]:
        for f in batch_files:
            print(f, number)
            print(files[batch_size * counter:batch_size * (counter+1)])
            # if success==batch_size:
            #     break
            try:
                print("loading")
                image = load_array(f, dim=pad)
                print(sys.getsizeof(image))
                image = image[:]
                print(type(image))
                print(sys.getsizeof(image))
                print(image.shape)
            except:
                print("couldn't load")
            if print_padded:
                print('loaded %s %s' %(f,number))
            padded_image = np.zeros((pad, 512, 32, 32))
            len_image = min(len(image), pad)
            # print(len_image)
            # image = np.expand_dims(np.expand_dims(image[:512], 1),0)
            # print(image.shape)
            # print(len_image)

            padded_image[:len_image, :, :, :] = image[:pad]
            if print_padded:
                print('padded %s %s' %(f,number))
            image_list.append(padded_image)
            patient_id = os.path.basename(os.path.dirname(f))
            success += 1
            try:
                is_cancer = train_csv_table.loc[train_csv_table['id'] == patient_id]['cancer'].values[0]

                if is_cancer == 0:
                    mask = [1, 0]
                    # mask = [1]
                else:
                    mask = [0, 1]
                    # mask = [0]
            except:
                # print('Problem with %s' % patient_id)
                mask = [0.5, 0.5]
            mask_list.append(mask)
        counter += 1
        mask_list = np.array(mask_list)
        image_list = np.array(image_list)
        del padded_image, image
        if print_padded:
            print(len(image_list))
            print(image_list[0].shape)
        yield image_list, mask_list
        del image_list, mask_list

        if counter== number_of_batches:
            random.shuffle(files)
            counter = 0
            failed = 0


def batch_generator_npz(files, train_csv_table, batch_size, pad=600, print_padded=False, number=1):
    print('new generator created')
    number_of_batches = np.ceil(len(files)/batch_size)
    counter = 0
    random.shuffle(files)
    # next_files = copy.deepcopy(files)
    # failed = 0

    while True:
        # batch_files = files[batch_size*counter+failed:batch_size*(counter+1)+failed]
        image_list = []
        mask_list = []
        success = 0
        problem_files=[]
        batch_files = files[batch_size * counter:batch_size * (counter+1)]
        # for f in files[batch_size*counter+failed:]:
        for f in batch_files:
            if print_padded:
                print(f, number)
                print(files[batch_size * counter:batch_size * (counter+1)])
            # if success==batch_size:
            #     break
            try:
                if print_padded:
                    print("loading")
                image = np.load(f)
                if print_padded:
                    print(sys.getsizeof(image))
                image = image[:pad]
                if print_padded:
                    print(type(image))
                    print(sys.getsizeof(image))
                    print(image.shape)
            except:
                print("couldn't load")
            if print_padded:
                print('loaded %s %s' %(f,number))
            padded_image = np.zeros((pad, 512, 32, 32))
            len_image = min(len(image), pad)
            # print(len_image)
            # image = np.expand_dims(np.expand_dims(image[:512], 1),0)
            # print(image.shape)
            # print(len_image)

            padded_image[:len_image, :, :, :] = image[:pad]
            if print_padded:
                print('padded %s %s' %(f,number))
            image_list.append(padded_image)
            patient_id = os.path.basename(os.path.dirname(f))
            success += 1
            try:
                is_cancer = train_csv_table.loc[train_csv_table['id'] == patient_id]['cancer'].values[0]

                if is_cancer == 0:
                    mask = [1, 0]
                    # mask = [1]
                else:
                    mask = [0, 1]
                    # mask = [0]
            except:
                # print('Problem with %s' % patient_id)
                mask = [0.5, 0.5]
            mask_list.append(mask)
        counter += 1
        mask_list = np.array(mask_list)
        image_list = np.array(image_list)
        del padded_image, image
        if print_padded:
            print(len(image_list))
            print(image_list[0].shape)
        yield image_list, mask_list
        del image_list, mask_list

        if counter== number_of_batches:
            random.shuffle(files)
            counter = 0
            failed = 0