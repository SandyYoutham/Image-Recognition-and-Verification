import os
import cv2
import time
import random
import numpy as np

import tensorflow as tf
from keras.applications.inception_v3 import preprocess_input
from keras import backend, layers, metrics
from keras.optimizers import Adam
from keras.applications import Xception
from keras.models import Model, Sequential
from keras.utils import plot_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Setting random seeds to enable consistency while testing.
random.seed(5)
np.random.seed(5)
tf.random.set_seed(5)

Train_ROOT = "/kaggle/input/cv-recog/Training Data"
Val_ROOT = "/kaggle/input/cv-recog/Validation Data"


def read_image(root, index, target_size=(128, 128)):
    path = os.path.join(root, index[0], index[1])
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)  # Resize to the desired target size
    return image


def create_list(directory):
    folders = os.listdir(directory)
    num_train = int(len(folders))

    random.shuffle(folders)

    List = {}

    # Creating Train-list
    for folder in folders[:num_train]:
        num_files = len(os.listdir(os.path.join(directory, folder)))
        List[folder] = num_files

    print("Length of training list:", len(List))
    return List


train_list = create_list(Train_ROOT)
val_list = create_list(Val_ROOT)


def create_triplets(directory, folder_list):
    triplets = []
    folders = list(folder_list.keys())
    for folder in folders:
        path = os.path.join(directory, folder)
        files = list(os.listdir(path))[:]
        num_files = len(files)

        for i in range(1, num_files):
            for j in range(i + 1, num_files + 1):

                anchor = (folder, f"web{i}.png")
                positive = (folder, f"web{j}.png")

                neg_folder = folder
                while neg_folder == folder:
                    neg_folder = random.choice(folders)

                neg_file = random.randint(1, folder_list[neg_folder])
                negative = (neg_folder, f"web{neg_file}.png")
                triplets.append((anchor, positive, negative))

    random.shuffle(triplets)
    return triplets


train_triplet = create_triplets(Train_ROOT, train_list)
val_triplet = create_triplets(Val_ROOT, val_list)

print("Number of training triplets:", len(train_triplet))
print("Number of testing triplets :", len(val_triplet))

print("\nExamples of triplets:")
for i in range(5):
    print(train_triplet[i])


def get_batch(root, triplet_list, batch_size=256, preprocess=True):
    batch_steps = len(triplet_list) // batch_size

    for i in range(batch_steps + 1):
        anchor = []
        positive = []
        negative = []

        j = i * batch_size
        while j < (i + 1) * batch_size and j < len(triplet_list):
            a, p, n = triplet_list[j]

            anchor.append(read_image(root, a))
            positive.append(read_image(root, p))
            negative.append(read_image(root, n))
            j += 1

        anchor = np.array(anchor)
        positive = np.array(positive)
        negative = np.array(negative)

        if preprocess:
            anchor = preprocess_input(anchor)
            positive = preprocess_input(positive)
            negative = preprocess_input(negative)

        yield ([anchor, positive, negative])


def get_encoder(input_shape):
    """ Returns the image encoding model """

    pretrained_model = Xception(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
        pooling='avg',
    )

    for i in range(len(pretrained_model.layers) - 27):
        pretrained_model.layers[i].trainable = False

    encode_model = Sequential([
        pretrained_model,
        layers.Flatten(),
        #         layers.Dense(512, activation='relu'),
        #         layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # define a variable used in normalization
    ], name="Encode_Model")
    return encode_model


class DistanceLayer(layers.Layer):
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


def get_siamese_network(input_shape=(128, 128, 3)):
    encoder = get_encoder(input_shape)

    # Input Layers for the images
    anchor_input = layers.Input(input_shape, name="Anchor_Input")
    positive_input = layers.Input(input_shape, name="Positive_Input")
    negative_input = layers.Input(input_shape, name="Negative_Input")

    ## Generate the encodings (feature vectors) for the images
    encoded_a = encoder(anchor_input)
    encoded_p = encoder(positive_input)
    encoded_n = encoder(negative_input)

    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    distances = DistanceLayer()(
        encoded_a,
        encoded_p,
        encoded_n
    )

    # Creating the Model
    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=distances,
        name="Siamese_Network"
    )
    return siamese_network


siamese_network = get_siamese_network()
siamese_network.summary()


class SiameseModel(Model):
    # Builds a Siamese model based on a base-model
    def __init__(self, siamese_network, margin=1.0):
        super(SiameseModel, self).__init__()

        self.margin = margin
        self.siamese_network = siamese_network
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape get the gradients when we compute loss, and uses them to update the weights
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # Get the two distances from the network, then compute the triplet loss
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics so the reset_states() can be called automatically.
        return [self.loss_tracker]


siamese_model = SiameseModel(siamese_network)

optimizer = Adam(learning_rate=1e-3, epsilon=1e-01)
siamese_model.compile(optimizer=optimizer)


def validate_on_triplets(batch_size=256):
    pos_scores, neg_scores = [], []

    for data in get_batch(Val_ROOT, val_triplet, batch_size=batch_size):
        prediction = siamese_model.predict(data)
        pos_scores += list(prediction[0])
        neg_scores += list(prediction[1])

    accuracy = np.sum(np.array(pos_scores) < np.array(neg_scores)) / len(pos_scores)
    ap_mean = np.mean(pos_scores)
    an_mean = np.mean(neg_scores)
    ap_stds = np.std(pos_scores)
    an_stds = np.std(neg_scores)

    print(f"Accuracy on validation = {accuracy:.5f}")
    return (accuracy, ap_mean, an_mean, ap_stds, an_stds)


save_all = False
epochs = 15
batch_size = 128
max_acc = 0
train_loss = []
test_metrics = []
train_time = time.time()
correct_predictions = 0
total_samples = 0

for epoch in range(1, epochs + 1):
    t = time.time()
    Tpos_scores, Tneg_scores = [], []
    # Training the model on train data
    epoch_loss = []
    for data in get_batch(Train_ROOT, train_triplet, batch_size=batch_size):
        loss = siamese_model.train_on_batch(data)
        epoch_loss.append(loss)
        prediction = siamese_model.predict(data)
        Tpos_scores += list(prediction[0])
        Tneg_scores += list(prediction[1])

    training_accuracy = np.sum(np.array(Tpos_scores) < np.array(Tneg_scores)) / len(Tpos_scores)

    epoch_loss = sum(epoch_loss) / len(epoch_loss)
    train_loss.append(epoch_loss)
    print(f"\nEPOCH: {epoch} \t (Epoch done in {int(time.time() - t)} sec)")
    print(f"Training Accuracy = {training_accuracy:.5f}, Loss on train = {epoch_loss:.5f}")

    # Testing the model on test data
    metric = validate_on_triplets(batch_size=batch_size)
    test_metrics.append(metric)
    accuracy = metric[0]

    # Saving the model weights
    if save_all or accuracy >= max_acc:
        siamese_model.save_weights("siamese_model")
        max_acc = accuracy

print(f"Total training time: {int(time.time() - train_time)} sec")

# Saving the model after all epochs run
siamese_model.save_weights("siamese_model-final")


def extract_encoder(model):
    encoder = get_encoder((128, 128, 3))
    i = 0
    for e_layer in model.layers[0].layers[3].layers:
        layer_weight = e_layer.get_weights()
        encoder.layers[i].set_weights(layer_weight)
        i += 1
    return encoder


encoder = extract_encoder(siamese_model)
encoder.save_weights("encoder")
encoder.summary()
def save_encoder_weights(encoder, weights_file):
    # Save weights to a pickle file
    with open(weights_file, "wb") as pickle_file:
        pickle.dump(encoder.get_weights(), pickle_file)

save_encoder_weights(encoder, "encoder_weights.pickle")


def classify_images(face_list1, face_list2, threshold=1.3):
    # Getting the encodings for the passed faces
    tensor1 = encoder.predict(face_list1)
    tensor2 = encoder.predict(face_list2)

    distance = np.sum(np.square(tensor1 - tensor2), axis=-1)
    prediction = np.where(distance <= threshold, 0, 1)
    return prediction


def ModelMetrics(pos_list, neg_list):
    true = np.array([0] * len(pos_list) + [1] * len(neg_list))
    pred = np.append(pos_list, neg_list)

    # Compute and print the accuracy
    print(f"\nAccuracy of model: {accuracy_score(true, pred) * 100:.2f}\n")

    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(true, pred)

    categories = ['Similar', 'Different']
    names = ['True Similar', 'False Similar', 'False Different', 'True Different']
    percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)


pos_list = np.array([])
neg_list = np.array([])
val_time = time.time()
for data in get_batch(Val_ROOT, val_triplet, batch_size=256):
    a, p, n = data
    pos_list = np.append(pos_list, classify_images(a, p))
    neg_list = np.append(neg_list, classify_images(a, n))
    break

ModelMetrics(pos_list, neg_list)
print(f"Total validation time: {int(time.time() - val_time)} sec")
