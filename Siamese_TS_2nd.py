import os
import cv2
import glob
import pickle
import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.applications import Xception
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


def read_image(path, target_size=(128, 128)):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, target_size)  # Resize to the desired target size
    return im


# data reading
data_folder = 'Data\\Product Recoginition'
data = []
data_label = []
if os.path.isdir(data_folder):
    o_folder = os.path.join(data_folder, 'Training Data')
    if os.path.exists(o_folder) and os.path.isdir(o_folder):
        for folder_name in range(1, 41):
            folder_name = str(folder_name)
            folder_path = os.path.join(o_folder, folder_name)
            for i in range(1, 3):
                path = os.path.join(folder_path, f"web{str(i)}.png")
                image = read_image(path)
                data.append(image)
                data_label.append(folder_name)

tested_img_path = 'Test Samples Recognition'
test_img = []
test_label = []
test_names = []
for Test_folder in os.listdir(tested_img_path):
    Test_folder = str(Test_folder)
    test_folder_path = os.path.join(tested_img_path, Test_folder)
    if os.path.isdir(test_folder_path):
        Test_image_files = glob.glob(os.path.join(test_folder_path, '*'))
        for Timage_file in Test_image_files:
            image = read_image(Timage_file)
            test_img.append(image)
            test_label.append(Test_folder)
            test_names.append(os.path.basename(Timage_file))

# data preprocessing
data = preprocess_input(np.array(data))
test = preprocess_input(np.array(test_img))

# load encoder
encoder = get_encoder((128, 128, 3))
load_encoder_weights(encoder, "encoder_weights.pickle")

# encode images
encoded_data = encoder(data)
encoded_test = encoder(test)

ID = []
for i, enc_test in enumerate(encoded_test):
    min_dis = 100000.0
    min_index = -1
    for j, enc_data in enumerate(encoded_data):
        distance = np.sum(np.square(enc_test - enc_data), axis=-1)
        if distance < min_dis:
            min_dis = distance
            min_index = j

    pred_ID = data_label[min_index] if min_dis < 1.6 else -1
    print(f"predicted image: {test_names[i]}, pred ID: {pred_ID}, actual: {test_label[i]}")
    ID.append(pred_ID)

print(f"accuracy score:{accuracy_score(test_label, ID)*100}%")
