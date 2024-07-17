import os
import cv2
import glob
import pickle
import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.applications import Xception
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from keras.applications.inception_v3 import preprocess_input

# helper functions
def get_encoder(input_shape):
    """ Returns the image encoding model """

    pretrained_model = Xception(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
        pooling='avg',
    )

    for j in range(len(pretrained_model.layers) - 27):
        pretrained_model.layers[j].trainable = False

    encode_model = Sequential([
        pretrained_model,
        layers.Flatten(),
        # layers.Dense(512, activation='relu'),
        # layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="Encode_Model")
    return encode_model


def load_encoder_weights(encoder, weights_file):
    # Load weights from a pickle file
    with open(weights_file, "rb") as pickle_file:
        loaded_weights = pickle.load(pickle_file)
        encoder.set_weights(loaded_weights)


def read_image(path, target_size=(512, 512)):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, target_size)  # Resize to the desired target size
    return im


tested_img_path = 'One Shot Recognition'
test_img = []
test_names = []
if os.path.isdir(tested_img_path):
    Test_image_files = glob.glob(os.path.join(tested_img_path, '*'))
    for Timage_file in Test_image_files:
        if os.path.basename(Timage_file).startswith('anchor'):
            anchor = read_image(Timage_file)
        else:
            image = read_image(Timage_file)
            test_img.append(image)
            test_names.append(os.path.basename(Timage_file))
# print(len(test_img))
# print(len(test_names))
plt.imshow(anchor)
plt.show()
anchor = preprocess_input(np.expand_dims(np.array(anchor), axis=0))
images = preprocess_input(np.array(test_img))

encoder = get_encoder((512, 512, 3))
load_encoder_weights(encoder, "encoder_weights.pickle")

# encode images
encoded_images = encoder.predict(images)
encoded_anchor = encoder.predict(anchor)

min_dis = 100000.0
for i, enc_im in enumerate(encoded_images):
    distance = np.sum(np.square(encoded_anchor - enc_im), axis=-1)
    if distance < min_dis:
        min_dis = distance
        most_similar = i

# print(f"min dis:{min_dis}")
if min_dis < 0.9:
    plt.imshow(test_img[most_similar])
    plt.show()
    print(test_names[most_similar])
else:
    print("there isn't any similar image")
