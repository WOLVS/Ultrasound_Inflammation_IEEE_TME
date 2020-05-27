"""
This kernel is used to classify aishufan signals.

Author: Shufan Yang, shufany@gmail.com
Date: 25/11/2019
"""

import keras
from keras.models import load_model
from keras.optimizers import Adam, Nadam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet201
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
import shutil
from tqdm import tqdm
from random import seed
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import click
import time
import json
from scipy.stats import binom

from sf_mltool.sf_mltool import Sf_mltool, sf_visualize_train
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

GENERAL_IMAGE_OFFSET = 5

MODEL_NAMES = {
    'NM': 'NASNetMobile.h5',
    'NL': 'NASNetLarge.h5',
    'IR2': 'InceptionResNetV2.h5',
    'XC': 'Xception.h5',
    'D21': 'DenseNet201.h5',
    'IV3': 'InceptionV3.h5',
    'SC': 'SimpleCNN.h5',
}

system_config = {
    "Author": "Shufan Yang",
    "Email": "shufan.yang@glasgow.ac.uk",
    "Version": "1.0.0",
    "first_model": "NM",
    "second_model": "IV3",
    "IR2_Accuracy": 0.96,
    "NL_Accuracy": 0.96,
    "XC_Accuracy": 0.96,
    "D21_Accuracy": 0.96,
    "IV3_Accuracy": 0.96,
    "crop_size": 256,
    "image_wh": 128,
    "first_thd_offset": 2,
    "first_confidence": 0.99,
    "second_thd_offset": 2,
    "second_confidence": 0.80,
    "class_indices": {
    },
}

launch_folder = Path(os.getcwd())
script_folder = Path(os.path.dirname(os.path.realpath(__file__)))


def create_tv_list(folder, ext='jpg', split=0.20):
    """
    This function get all file names from the folder and creates two lists for
    training and validation

    If a file name contains 'FLM', its classification mark is '1'; If a file name
    contains 'NON', its classification mark is '0'.

    Example:
    train_list, valid_list = create_tv_list(source_folder.joinpath('label'))
    train_df = pd.DataFrame(train_list,columns=['fname','class'])

    :param folder: The folder holding all images
    :param ext: image file extension
    :param split: train, validation split ratio
    :return: train_list, valid_list
    """
    # Get a file list and classification
    labels = set()
    flist = []
    files = Path(folder).glob(f'**/*.{ext}')
    labels = ['NON', 'FLM']
    for i in files:
        temp = i.stem[0:3]
        if temp == 'NON' or temp == 'FLM':
            flist.append([str(i), temp])

    train_list = []
    valid_list = []
    for i in labels:
        temp = [element for element in flist if element[1] == i]
        temp = shuffle(temp, random_state=seed())
        tl, vl = train_test_split(temp, test_size=split)
        train_list = train_list + tl
        valid_list = valid_list + vl

    return train_list, valid_list


def create_ml_list(folder, ext='jpg', split=0.33):
    """
    This function get all file names from the folder and creates three lists for
    machine learning (ml).

    If a file name contains 'FLM', its classification mark is '1'; If a file name
    contains 'NON', its classification mark is '0'.

    Example:
    train_list, valid_list, test_list = create_ml_list(source_folder.joinpath('label'))
    train_df = pd.DataFrame(train_list,columns=['fname','class'])

    :param folder: The folder holding all images
    :param ext: image file extension
    :param split: train, validation, test dataset split ratio
    :return: train_list, valid_list, test_list
    """
    # Get a file list and classification
    flist = []
    files = Path(folder).glob(f'**/*.{ext}')
    for i in files:
        if 'FLM' in i.stem:
            flist.append([str(i), '1'])
        elif 'NON' in i.stem:
            flist.append([str(i), '0'])
        else:
            pass
    pass

    # shuffle flist
    flist = shuffle(flist, random_state=seed())
    train_list, temp_list = train_test_split(flist, test_size=split)
    test_list, valid_list = train_test_split(temp_list, test_size=split)

    #print(f'flist size: {len(flist)}, train size: {len(train_list)}, valid size: {len(valid_list)}, test size: {len(test_list)}')
    return train_list, valid_list, test_list


def _draw_contours(fname, thdl=80, thdh=240):
    """
    draw contours of image
    :param fname: image file name
    :param thdl: contour detection low threshold
    :param thdh: contour detection high threshold
    :return: None
    """
    ifd = cv2.imread(str(fname))
    cv2.imwrite(str(script_folder.joinpath('test.jpg')), ifd)

    gray = cv2.cvtColor(ifd, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    blur[0:10, :] = 0
    blur[(ifd.shape[0]-10):ifd.shape[0], :] = 0

    _, binary = cv2.threshold(blur, thdl, thdh, cv2.THRESH_BINARY)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image=ifd,
                     contours=contours,
                     contourIdx=-1,
                     color=(0, 200, 25),
                     thickness=5)

    cv2.imshow('Contours', ifd)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_ultrasound_image(folder, cl, fname):
    """
    This function read the image in tiff format. It then extracts
    the image into smaller sub images to be trained, validated, or tested.

    :note A: folder should exist, function could throw exception
    :param folder: destination folder where new extracted files are saved
    :param cl: class label to be added on extracted image name as prefix
    :param fname: marked image file name
    :return: extracted images in folder
    """
    images = []
    extracted_image_wh = system_config['image_wh']

    # Create new image
    ifd = cv2.imread(str(fname), cv2.IMREAD_COLOR)
    ws = ifd.shape[0]
    hs = ifd.shape[1]

    # Find contours
    gray = cv2.cvtColor(ifd, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    blur[0:10, :] = 0
    blur[(ws-10):ws, :] = 0
    # The operator should be &, do not use &&
    mask = (blur > 40) & (blur < 220)

    # Debug only
    if False:
        # Strip off edges
        offset = 10
        stipped = blur[offset:hs-offset, offset:ws-offset]

    _, binary = cv2.threshold(blur, 40, 240, cv2.THRESH_BINARY)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for debug only
    if False:
        cv2.drawContours(image=npimage,
                         contours=contours,
                         contourIdx=-1,
                         color=(0, 200, 25),
                         thickness=5)
        cv2.imshow('test', npimage)
        cv2.waitKey(0)

    tl_x = ws
    tl_y = hs
    br_x = 0
    br_y = 0
    for i in contours:
        (x, y, w, h) = cv2.boundingRect(i)
        s = x + w
        t = y + h
        tl_x = x if (tl_x > x) else tl_x
        tl_y = y if (tl_y > y) else tl_y
        br_x = s if (br_x < s) else br_x
        br_y = t if (br_y < t) else br_y

    # Fix width
    fixed_width = 400
    brw = br_x - tl_x
    brh = br_y - tl_y
    if (brh > fixed_width):
        # Cropping height is too big, change it smaller.
        offset = (brh - fixed_width)/2
        tl_y = tl_y + offset
        br_y = br_y - offset
    else:
        #TODO: tl_y < 0 is not checked
        offset = (fixed_width - brh)/2
        tl_y = tl_y - offset
        br_y = br_y + offset

    # Crop it to designated size
    num = int((br_x - tl_x)/fixed_width)
    for i in range(num):
        box = [tl_x + i*fixed_width, tl_y, tl_x + (i + 1)*fixed_width, br_y]
        if box[2] > br_x:
            box[2] = br_x
            box[0] = box[2] - fixed_width

        ofd = ifd[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
        ofdr = ofd[:, :, 0]
        ofdr[(ofdr < 150) & (ofdr > 130)] = 0
        # ofd[:, :, 0] = 0
        ofd[:, :, 0] = ofdr

        # Save it
        idx = 1
        ofname = folder.joinpath(f'{cl}_{fname.stem}_{idx}.jpg')
        while (ofname.is_file()):
            idx = idx + 1
            ofname = folder.joinpath(f'{cl}_{fname.stem}_{idx}.jpg')

        cvofd = cv2.resize(ofd, (extracted_image_wh, extracted_image_wh), interpolation=cv2.INTER_NEAREST)
        images.append(cvofd)

        cv2.imwrite(str(ofname), cvofd)

    np_images_shape = (len(images), extracted_image_wh, extracted_image_wh, 3)
    np_images = np.zeros(np_images_shape, dtype=np.uint8)
    for i, image in enumerate(images):
        np_images[i] = image

    return np_images


def extract_images(dst_folder, src_folder, re, cl, extracted_image_wh=128):
    """
    Extract all marked images in src_folder and save them into dst_folder.

    Example:

    src = Path('C:/Users/LiShoujie/Desktop/ml/ml/ImgTree/0614-1')
    dst = Path('C:/Users/LiShoujie/Desktop/ml/ml/ImgTree/test')

    extract_images(dst, src)

    :param dst_folder: full path destination folder for extracted images
    :param src_folder: full path source folder where keeps the original images
    :param re: src_folder name pattern
    :param cl: class label to be added on new image file name as prefix
    :param extracted_image_wh: extracted image size
    :param ext: image file extension
    :return: None
    """
    if extracted_image_wh != system_config['image_wh']:
        system_config['image_wh'] = extracted_image_wh
        save_config()

    # Remove previous files
    shutil.rmtree(str(dst_folder), ignore_errors=True)
    dst_folder.mkdir(exist_ok=True)

    # Get all files
    files = list(Path(src_folder).glob(re))
    total = len(files)

    # extract one by one
    with tqdm(total=total) as pbar:
        for i in files:
            try:
                extract_ultrasound_image(dst_folder, cl, i)
            except:
                print(f'file extraction error: {str(i)}')
                pass
            pbar.set_description(str(i))
            pbar.update(1)


def data_generator():
    """
    This augments image in real time.
    :return: image data generator
    """
    return ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=True,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0.0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.0,
        # set range for random zoom
        zoom_range=0.2,
        # set range for random channel shifts
        channel_shift_range=0.0,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=False,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def simple_cnn(input_shape=None, classes=0):
    """
    Simple CNN to train aishufan network
    :param input_shape:
    :return:
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    return model


def unet_cnn(input_shape=None, classes=0):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Reshape((14, 14, 8)))

    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3)))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3)))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    return model


def aishufan_train(folder, batch_size, epoch_size, model_name):
    """
    Train network with the parameters specified.

    :param folder: image folder for training
    :param batch_size: training batch size
    :param epoch_size: training epoch size
    :param model_name: IR2, InceptionResNetV2; NL, NASNetLarge; NM, NASNetLarge
    :return: None
    """
    image_wh = system_config['image_wh']

    image_size = (image_wh, image_wh)
    image_shape= (image_wh, image_wh, 3)

    train_list, valid_list = create_tv_list(folder)
    print(f'Train size: {len(train_list)}, valid size: {len(valid_list)}')

    train_df = pd.DataFrame(train_list, columns=['fname', 'class'])
    valid_df = pd.DataFrame(valid_list, columns=['fname', 'class'])

    model = None
    if 'NM' in model_name:
        model_name = 'NM'
        model = NASNetMobile(include_top=True,
                             weights=None,
                             input_tensor=None,
                             input_shape=image_shape,
                             pooling='max',
                             classes=2)
    elif 'NL' in model_name:
        model_name = 'NL'
        model = NASNetLarge(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=image_shape,
                            pooling='max',
                            classes=2)
    elif 'XC' in model_name:
        model_name = 'XC'
        model = Xception(include_top=True,
                         weights=None,
                         input_tensor=None,
                         input_shape=image_shape,
                         pooling='max',
                         classes=2)
    elif 'D21' in model_name:
        model_name = 'D21'
        model = DenseNet201(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=image_shape,
                            pooling='max',
                            classes=2)
    elif 'IV3' in model_name:
        model_name = 'IV3'
        model = InceptionV3(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=image_shape,
                            pooling='max',
                            classes=2)
    elif 'SC' in model_name:
        model_name = 'SC'
        model = unet_cnn(input_shape=image_shape,
                           classes=2)
    else:
        model_name = 'IR2'
        model = InceptionResNetV2(include_top=True,
                                  weights=None,
                                  input_tensor=None,
                                  input_shape=image_shape,
                                  pooling='max',
                                  classes=2)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()

    # Image generator does data augmentation:
    datagen = data_generator()

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=folder,
        x_col="fname",
        y_col="class",
        class_mode="categorical",
        target_size=image_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False)

    valid_gen = datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=folder,
        x_col="fname",
        y_col="class",
        class_mode="categorical",
        target_size=image_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False)

    # Save class indices
    system_config['class_indices'] = train_gen.class_indices
    save_config()

    # Prepare model model saving directory.
    save_dir = Path(os.path.dirname(os.path.realpath(__file__))).joinpath('models')
    if not save_dir.is_dir():
        save_dir.mkdir(exist_ok=True)
    filepath = f'{str(save_dir)}/{MODEL_NAMES[model_name]}'
    print(f'{filepath}\n')

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # Fit the model on the batches generated by datagen.flow().
    steps_per_epoch = int(len(train_list)/batch_size)
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen,
        validation_steps=steps_per_epoch,
        epochs=epoch_size,
        use_multiprocessing=False,
        verbose=1,
        workers=4,
        callbacks=callbacks)

    # Score trained model.
    scores = model.evaluate_generator(generator=valid_gen, steps=steps_per_epoch, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # Save score in configuration file
    system_config[f'{model_name}_Accuracy'] = scores[1]
    save_config()

    return history


def autoencoder_train(folder, batch_size, epoch_size, model_name):
    """
    Autoencoding, inherently UNET, is a data compression algorithm where the compression and decompression functions are:
    - data specific, ie, only compress data similar to what they have been trained on
    - lossy, ie, decompressed output will be degraded
    - learned automatically from examples.

    Two practical applications of autoencoders are data removal and dimensionality reduction

    There is an implementation from scikit-learn:
    http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    :param folder: image folder for training
    :param batch_size: training batch size
    :param epoch_size: training epoch size
    :param model_name: IR2, InceptionResNetV2; NL, NASNetLarge; NM, NASNetLarge
    :return: None
    """
    image_wh = system_config['image_wh']

    image_size = (image_wh, image_wh)
    image_shape= (image_wh, image_wh, 1)

    train_list, valid_list = create_tv_list(folder)
    print(f'Train size: {len(train_list)}, valid size: {len(valid_list)}')

    train_df = pd.DataFrame(train_list, columns=['fname', 'class'])
    valid_df = pd.DataFrame(valid_list, columns=['fname', 'class'])

    model = None
    if 'NM' in model_name:
        model_name = 'NM'
        model = NASNetMobile(include_top=True,
                             weights=None,
                             input_tensor=None,
                             input_shape=image_shape,
                             pooling='max',
                             classes=2)
    elif 'NL' in model_name:
        model_name = 'NL'
        model = NASNetLarge(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=image_shape,
                            pooling='max',
                            classes=2)
    elif 'XC' in model_name:
        model_name = 'XC'
        model = Xception(include_top=True,
                         weights=None,
                         input_tensor=None,
                         input_shape=image_shape,
                         pooling='max',
                         classes=2)
    elif 'D21' in model_name:
        model_name = 'D21'
        model = DenseNet201(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=image_shape,
                            pooling='max',
                            classes=2)
    elif 'IV3' in model_name:
        model_name = 'IV3'
        model = InceptionV3(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=image_shape,
                            pooling='max',
                            classes=2)
    elif 'SC' in model_name:
        model_name = 'SC'
        model = simple_cnn(input_shape=image_shape,
                           classes=2)
    else:
        model_name = 'IR2'
        model = InceptionResNetV2(include_top=True,
                                  weights=None,
                                  input_tensor=None,
                                  input_shape=image_shape,
                                  pooling='max',
                                  classes=2)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()

    # Image generator does data augmentation:
    datagen = data_generator()

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=folder,
        x_col="fname",
        y_col="class",
        class_mode="categorical",
        target_size=image_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False)

    valid_gen = datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=folder,
        x_col="fname",
        y_col="class",
        class_mode="categorical",
        target_size=image_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False)

    # Prepare model model saving directory.
    save_dir = Path(os.path.dirname(os.path.realpath(__file__))).joinpath('models')
    if not save_dir.is_dir():
        save_dir.mkdir(exist_ok=True)
    filepath = f'{str(save_dir)}/{MODEL_NAMES[model_name]}'
    print(f'{filepath}\n')

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # Fit the model on the batches generated by datagen.flow().
    steps_per_epoch = int(len(train_list)/batch_size)
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen,
        validation_steps=steps_per_epoch,
        epochs=epoch_size,
        use_multiprocessing=False,
        verbose=1,
        workers=4,
        callbacks=callbacks)

    # Score trained model.
    scores = model.evaluate_generator(generator=valid_gen, steps=steps_per_epoch, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # Save score in configuration file
    system_config[f'{model_name}_Accuracy'] = scores[1]
    save_config()

    return history


def transfer_learn(folder, batch_size, epoch_size, model_name):
    """
    Transfer learning is a procedure with pre-trained models.

    :param folder: image folder for training
    :param batch_size: training batch size
    :param epoch_size: training epoch size
    :param model_name: IR2, InceptionResNetV2; NL, NASNetLarge; NM, NASNetLarge
    :return: None
    """
    image_wh = system_config['image_wh']

    image_size = (image_wh, image_wh)
    image_shape= (image_wh, image_wh, 1)

    train_list, valid_list = create_tv_list(folder)
    print(f'Train size: {len(train_list)}, valid size: {len(valid_list)}')

    train_df = pd.DataFrame(train_list, columns=['fname', 'class'])
    valid_df = pd.DataFrame(valid_list, columns=['fname', 'class'])

    model_fname = f'models/{MODEL_NAMES[model_name]}'
    model_fname = script_folder.joinpath(model_fname)
    model = load_model(str(model_fname), compile=False)

    # Set first 50 layer non trainable
    for layer in model.layers[50:]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()

    # Image generator does data augmentation:
    datagen = data_generator()

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=folder,
        x_col="fname",
        y_col="class",
        class_mode="categorical",
        target_size=image_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False)

    valid_gen = datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=folder,
        x_col="fname",
        y_col="class",
        class_mode="categorical",
        target_size=image_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False)

    # Prepare model model saving directory.
    save_dir = Path(os.path.dirname(os.path.realpath(__file__))).joinpath('models')
    if not save_dir.is_dir():
        save_dir.mkdir(exist_ok=True)
    filepath = f'{str(save_dir)}/{MODEL_NAMES[model_name]}'

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # Fit the model on the batches generated by datagen.flow().
    steps_per_epoch = int(len(train_list)/batch_size)
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen,
        validation_steps=steps_per_epoch,
        epochs=epoch_size,
        use_multiprocessing = False,
        verbose=1,
        workers=4,
        callbacks=callbacks)

    # Score trained model.
    scores = model.evaluate_generator(generator=valid_gen, steps=steps_per_epoch, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # Save score in configuration file
    system_config[f'{model_name}_Accuracy'] = scores[1]
    save_config()

    return history


def load_config(fname='aishufan.json'):
    """
    Configuration file is in JSON format.

    :param fname: Configuration file name
    :return: system configuration
    """
    config = {}
    try:
        with open(script_folder.joinpath(fname)) as f:
            config = json.load(f)
    except:
        # Create default configuration file
        pass
    finally:
        print(config)

    return config


def save_config(fname='aishufan.json'):
    """
    save configuration file is in JSON format.

    :param fname: Configuration file name
    :return:
    """
    try:

        with open(script_folder.joinpath(fname), 'w', encoding='utf-8') as f:
            json.dump(system_config, f, ensure_ascii=False, indent=4)
    except:
        # Create default configuration file
        pass


def decision_threshold(overall=0.99, accuracy=0.95, samples=1000):
    """
    Calculate the decision threshold. This is based on Binomial Distribution.
    :param overall: Certainty we can sure the image is FLM
    :param accuracy: Accuracy of predicting NON is NON
    :param samples: How many sub images in total
    :return: integer value; if more than this value of sub images are FLM, this image is FLM
    """
    if samples == 0:
        return 0
    else:
        k = binom.ppf(overall, samples, 1-accuracy)
        return int(k)


def double_check(fname, model_first, data_generator):
    """
    Double check whether the input image is FLM or not. The check, or infer is
    done with 2 models.
    :param fname: image file name
    :param model_first: The first neural network model
    :param data_generator: data generator for neural network model
    :return: image inference label and its description
    """
    # The following code is for storing files to check manually
    check_dir = script_folder.joinpath('check')
    first_dir = check_dir.joinpath('first')
    second_dir = check_dir.joinpath('second')

    # create debug folder
    debug_folder = script_folder.joinpath('debug')
    shutil.rmtree(str(debug_folder), ignore_errors=True)
    debug_folder.mkdir(exist_ok=True)

    # Predict with first model
    images = extract_ultrasound_image(debug_folder, 'debug', fname)
    gen = data_generator.flow(images, shuffle=False)
    probabilities = model_first.predict_generator(generator=gen, steps=len(gen), verbose=1)

    # TODO: class_index might returns a low possibility class.
    pred = np.argmax(probabilities, axis=-1)
    class_index = max(pred)

    class_list = list(system_config['class_indices'].keys())
    rdes1 = 'first inference: '
    for row in probabilities:
        for idx, col in enumerate(row):
            des = f'{class_list[idx]}: {col}; '
            rdes1 += des

    # Save first model data for later check.
    ts = time.time()
    for i in range(images.shape[0]):
        ofname = first_dir.joinpath(f'{ts}_{class_list[pred[i]]}_{i}.jpg')
        cv2.imwrite(str(ofname), images[i])

    # Not used any more
    '''
    rdes2 = 'second inference: '
    if count_first < 1 and False:
        # Predict again with second model
        gen = data_generator.flow(images, shuffle=False)
        probabilities = model_second.predict_generator(generator=gen, steps=len(gen), verbose=0)
        pred = np.argmax(probabilities, axis=-1)
        count_second = np.count_nonzero(pred)

        for row in probabilities:
            for idx, col in enumerate(row):
                des = f'{idx}: {col}; '
                rdes2 += des

        # for manual check
        for i in range(images.shape[0]):
            ofname = second_dir.joinpath(f'{ts}_FLM_{i}.jpg') if pred[i] == 1 else second_dir.joinpath(
                f'{ts}_NON_{i}.jpg')
            cv2.imwrite(str(ofname), images[i])

        model_name = system_config['second_model']
        threshold_second = 1
        print(f'Second detected regions: {count_second}; decision threshold: {threshold_second}')
        if count_second < threshold_second:
            result = False
        else:
            result = True
    else:
        result = True
    '''

    return (class_list[class_index], rdes1)


def infer_batch(src_folder, re, result_file, debug=False):
    """
    Infer the class of an iPad from captured image. For one image, 2 different
    models are used to predict its class. This ensures prediction accuracy.

    :param src_folder: source image folder.
    :param re: image file name regular expression
    :param result_file: infer result file
    :param debug: Debug flag
    :return: None
    """
    check_dir = script_folder.joinpath('check')
    if not check_dir.is_dir():
        check_dir.mkdir(exist_ok=True)

    first_dir = check_dir.joinpath('first')
    if not first_dir.is_dir():
        first_dir.mkdir(exist_ok=True)

    # Load model, probably find a latest file in saved model folder
    model_indexes = ('NM', 'NM')
    # first prediction model
    model_fname = f'models/{MODEL_NAMES[model_indexes[0]]}'
    model_fname = script_folder.joinpath(model_fname)
    model_first = load_model(str(model_fname), compile=False)

    # Image generator does data augmentation
    datagen = data_generator()

    # Remove previous infer result
    if result_file.is_file():
        result_file.unlink()

    # Get all files
    files = list(Path(src_folder).glob(re))
    total = len(files)

    # infer one by one
    fname = script_folder.joinpath('result.csv')
    with open(str(fname), 'w') as ofd:
        with tqdm(total=total) as pbar:
            for i in files:
                # Wait for signal from external device or a socket server?
                try:
                    result = double_check(i, model_first, datagen)
                    ofd.write(f'{result[0]}, "{result[1]}", {str(i)}\r\n')
                except:
                    pass

                pbar.set_description(str(i))
                pbar.update(1)


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version='1.0.0')
def aishufan_ic_cli():
    pass


@aishufan_ic_cli.command()
@click.option('-f', '--file-name', required=True, help='image file name, relative to current folder')
@click.option('-l', '--low-thd', default=80, type=int, show_default=True, help='low contour threshold')
@click.option('-h', '--high-thd', default=240, type=int, show_default=True, help='low contour threshold')
def draw_contours(**kwargs):
    fname = script_folder.joinpath(kwargs['file_name'])
    lthd = kwargs['low_thd']
    hthd = kwargs['high_thd']
    _draw_contours(fname, lthd, hthd)


@aishufan_ic_cli.command()
@click.option('-f', '--file-name', required=True, help='image file name, relative to current folder')
@click.option('-e', '--epoch', default=200, help='epoch count to draw')
def visualize_train(**kwargs):
    fname = script_folder.joinpath(kwargs['file_name'])
    epoch = kwargs['epoch']
    sf_visualize_train(fname, epoch)


@aishufan_ic_cli.command()
@click.option('-d', '--dst-folder', default='extract', help='destination image folder, relative to current folder')
@click.option('-s', '--src-folder', required=True, help='source image folder, relative to current folder')
@click.option('-r', '--re-pattern', help='file name pattern')
@click.option('-l', '--class-label', default='C0', help='class label to a prefix of extracted image name')
@click.option('-i', '--image-size', default=128, type=int, show_default=True, help='extracted image size resized from cropped image size')
def extract(**kwargs):
    # extract -d extract -s contour -r **/*Segmented*/*.tif
    src = script_folder.joinpath(kwargs['src_folder'])
    dst = script_folder.joinpath(kwargs['dst_folder'])
    re = kwargs['re_pattern']
    cl = kwargs['class_label']
    extract_images(dst, src, re, cl, kwargs['image_size'])


@aishufan_ic_cli.command()
@click.option('-s', '--image-folder', default='extracted', help='image folder for training, relative to current folder')
@click.option('-b', '--batch-size', default=128, type=int, show_default=True, help='batch size')
@click.option('-e', '--epoch-size', default=500, type=int, show_default=True, help='epoch size')
@click.option('-m', '--model-name', default='IR2', show_default=True, help='IR2, NM, IV3, XC, D21')
def train(**kwargs):
    folder = script_folder.joinpath(kwargs['image_folder'])
    history = aishufan_train(folder,
                        kwargs['batch_size'],
                        kwargs['epoch_size'],
                        kwargs['model_name'])


@aishufan_ic_cli.command()
@click.option('-s', '--image-folder', default='aishufanExtracted', help='image folder for training, relative to current folder')
@click.option('-b', '--batch-size', default=128, type=int, show_default=True, help='batch size')
@click.option('-e', '--epoch-size', default=500, type=int, show_default=True, help='epoch size')
@click.option('-m', '--model-name', default='IR2', show_default=True, help='IR2, NM, IV3, XC, D21')
def transfer(**kwargs):
    folder = script_folder.joinpath(kwargs['image_folder'])
    history = transfer_learn(folder,
                             kwargs['batch_size'],
                             kwargs['epoch_size'],
                             kwargs['model_name'])


@aishufan_ic_cli.command()
@click.option('-s', '--src-folder', required=True, help='source image folder, relative to current folder')
@click.option('-r', '--re-pattern', help='file name pattern')
@click.option('-f', '--result-file', default='infer_result.csv', help='infer result')
@click.option('-d', '--debug', is_flag=True, default=False, show_default=True, help='debug information during infer')
def infer(**kwargs):
    src = script_folder.joinpath(kwargs['src_folder'])
    re = kwargs['re_pattern']
    fname = script_folder.joinpath(kwargs['result_file'])
    infer_batch(src, re, fname, kwargs['debug'])


@aishufan_ic_cli.command()
@click.option('-s', '--image-folder', required=True, help='image verification folder')
def verify(**kwargs):
    aishufan_verify(kwargs['image_folder'])


if __name__ == '__main__':
    # Load and merge configuration
    config = load_config()
    if config != system_config:
        system_config = {**system_config, **config}
        save_config()

    # Execute commands
    aishufan_ic_cli()
