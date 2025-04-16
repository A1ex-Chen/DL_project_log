'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

image classification on cifar10

train.py desc: loads data, trains and saves model, plots training metrics
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
import keras_model

import datetime

EPOCHS = 500
BS = 32

# get date ant time to save model
dt = datetime.datetime.today()
year = dt.year
month = dt.month
day = dt.day
hour = dt.hour
minute = dt.minute

"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 
training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains 
exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random 
order, but some training batches may contain more images from one class than another. Between them, the training 
batches contain exactly 5000 images from each class.
"""

#learning rate schedule

lr_scheduler = LearningRateScheduler(lr_schedule)

#optimizer
optimizer = tf.keras.optimizers.Adam()

#define data generator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    #brightness_range=(0.9, 1.2),
    #contrast_range=(0.9, 1.2),
    validation_split=0.2
)





if __name__ == "__main__":
    """load cifar10 data and trains model"""

    cifar_10_dir = 'cifar-10-batches-py'

    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        load_cifar_10_data(cifar_10_dir)

    print("Train data: ", train_data.shape)
    print("Train filenames: ", train_filenames.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)

    # Don't forget that the label_names and filesnames are in binary and need conversion if used.

    # display some random training images in a 25x25 grid
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot):
        for n in range(num_plot):
            idx = np.random.randint(0, train_data.shape[0])
            ax[m, n].imshow(train_data[idx])
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)
    plt.show()

    new_model = keras_model.resnet_v1_eembc()
    new_model.summary()

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(train_data)

    new_model.compile(
        optimizer=optimizer, loss='categorical_crossentropy', metrics='accuracy', loss_weights=None,
        weighted_metrics=None, run_eagerly=None )

    # fits the model on batches with real-time data augmentation:
    History = new_model.fit(datagen.flow(train_data, train_labels, batch_size=BS),
              steps_per_epoch=len(train_data) / BS, epochs=EPOCHS, callbacks=[lr_scheduler])

    plt.plot(np.array(range(EPOCHS)), History.history['loss'])
    plt.plot(np.array(range(EPOCHS)), History.history['accuracy'])
    plt.savefig('train_loss_acc.png')
    model_name = "trainedResnet.h5"
    new_model.save("trained_models/" + model_name)