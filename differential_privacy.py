import tensorflow as tf
print(tf.__version__)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# print(tf.config.list_physical_devices('GPU'))

import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
import os
import pathlib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical

# Dataset directory (Nanti ubah ikut path kat laptop kau)
dataset_dir=r"C:\Users\60192\OneDrive\Documents\MeID\datasets\cataract"

#class_names tu nanti ubah ikut dataset mcm skincancer ada 4 class so guna 0,1,2,3
def load_dataset(dataset_path):
    data = []
    labels = []
    class_names = ['0', '1']
    #class_names = ['0', '1', '2', '3']
    #class_names = ['normal', 'pneumonia']  # Define your class names

    for class_name in class_names:
        #class_path = dataset_path / class_name
        class_path=os.path.join(dataset_dir, class_name)
        #image_paths = list(class_path.glob("*.[jp][pn]g")) 
        class_path=Path(class_path)
        image_paths=(list(class_path.glob("*.[jp][pn]g"))) + list(class_path.glob("*.bmp"))+ list(class_path.glob("*.jpeg"))
        
        for image_path in image_paths:
            image = tf.io.read_file(str(image_path))
            
            
            image_extension = image_path.suffix.lower()
            if image_extension == '.jpeg' or image_extension == '.jpg':
                image = tf.image.decode_jpeg(image, channels=3)
            elif image_extension == '.png':
                image = tf.image.decode_png(image, channels=3)
            elif image_extension == '.bmp':
                image = tf.image.decode_bmp(image)
            else:
                raise ValueError(f"Unsupported image format: {image_extension}")
            
            
            
            #image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, (128, 128))  # Resize the image to the desired size 224->128->64
            data.append(image)
            labels.append(class_name)

    data = tf.stack(data)
    labels = tf.constant(labels)

    return data, labels

data, labels = load_dataset(dataset_dir)
custom_dataset = tf.data.Dataset.from_tensor_slices((data, labels))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(np.array(data), np.array(labels), test_size=0.2, random_state=42)

# Set validation ratio 
VALID_PERCENT = 0.1

split_on = int((1 - VALID_PERCENT) * len(x_train))

train_images = x_train[0:split_on,:,:]
train_labels = to_categorical(y_train)[0:split_on,:]

valid_images = x_train[split_on:,:,:]
valid_labels = to_categorical(y_train)[split_on:,:]

test_images = x_test
test_labels = to_categorical(y_test)

def preprocess(images):
  rescaled_images=images/255
  return rescaled_images

train_img=preprocess(train_images)
valid_img=preprocess(valid_images)
test_img=preprocess(test_images)

tf.get_logger().setLevel('ERROR')

print("Train Image Min:", train_img.min())
print("Train Image Max:", train_img.max())
print("Valid Image Min:", valid_img.min())
print("Valid Image Max:", valid_img.max())

epochs = 100
batch_size = 128

l2_norm_clip = 1.5 
noise_multiplier = 0.3 
num_microbatches = 1
learning_rate = 0.01 #0.00001

if batch_size % num_microbatches != 0:
  raise ValueError('Batch size should be an integer multiple of the number of microbatches')

#VGG16 MODEL
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras.layers import Input
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns

IMG_SIZE=(128,128)

# load base model
vgg16_weight_path = r"C:\Users\60192\OneDrive\Documents\MeID\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
vgg = VGG16(
    weights=vgg16_weight_path,
    include_top=False,
    input_shape=IMG_SIZE + (3, )
)

NUM_CLASSES = 2

vgg16 = Sequential()
# vgg16.add(Input(shape=(128, 128, 3)))
vgg16.add(vgg)
vgg16.add(layers.Dropout(0.3))
vgg16.add(layers.Flatten())
vgg16.add(layers.Dropout(0.5))
vgg16.add(layers.Dense(NUM_CLASSES, activation='softmax'))

vgg16.layers[0].trainable = False

optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=num_microbatches,
    learning_rate=learning_rate)

loss = tf.keras.losses.CategoricalCrossentropy(              
    from_logits=False, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

from keras.callbacks import EarlyStopping
early_stopping_callback = EarlyStopping(monitor='val_loss',patience=20)

vgg16.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

history=vgg16.fit(train_img, train_labels,
      epochs=epochs,
      validation_data=(valid_img, valid_labels),
      batch_size=batch_size,
      callbacks= [early_stopping_callback])

print("Learning stopped on epoch:", early_stopping_callback.stopped_epoch)

train_loss, train_acc = vgg16.evaluate(train_img, train_labels, verbose=2)
print('\nTrain accuracy:', train_acc)
print('Train loss:', train_loss)

test_loss, test_acc = vgg16.evaluate(test_img, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print('Test loss:', test_loss)

val_loss, val_acc = vgg16.evaluate(valid_img, valid_labels, verbose=2)
print('\nVal accuracy:', val_acc)
print('Val loss:', val_loss)

vgg16.save("dpCataract.h5")