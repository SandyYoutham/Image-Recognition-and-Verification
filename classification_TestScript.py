import os
import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


data_folder = 'Test Samples Classification'

# Initialize an empty list to store image paths
test_data = []
test_labels = []

# Iterate through each numbered folder
for folder_name in os.listdir(data_folder):
    folder_name = str(folder_name)
    folder_path = os.path.join(data_folder, folder_name)
    # Check if the folder is a directory
    if os.path.isdir(folder_path):
        # Check if the 'Train' folder exists within the numbered folder
            test_image_files = glob.glob(os.path.join(folder_path, '*'))
            for Vimage_file in test_image_files:
                # Read the image and append it to the training data list
                image1 = cv2.imread(Vimage_file)
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                image1 = cv2.resize(image1, (224, 224))
                image1 = np.expand_dims(image1, axis=-1)
                test_data.append(image1)
                test_labels.append(folder_name)


# -------------------------------------------------------------validation
kmeans_obj = pickle.load(open("kmeans.pickle", 'rb'))
num_of_clusters = pickle.load(open("num_of_clusters.pickle", 'rb'))
scaler = pickle.load(open("scaler.pickle", 'rb'))
svm = pickle.load(open("svm.pickle", 'rb'))
log = pickle.load(open("log.pickle", 'rb'))

sift = cv2.SIFT_create()
val_descriptor_list = []
for Vimg in test_data:
    kp, des = sift.detectAndCompute(Vimg, None)
    val_descriptor_list.append(des)

all_val_descriptors = np.vstack(val_descriptor_list)
val_predicted_labels = kmeans_obj.predict(all_val_descriptors)
val_mega_histogram = np.array([np.zeros(num_of_clusters) for i in range(len(test_data))])
old_count = 0
for i in range(len(test_data)):
    l = len(val_descriptor_list[i])
    for j in range(l):
        idx = val_predicted_labels[old_count + j]
        val_mega_histogram[i][idx] += 1
    old_count += l

vy_scalar = np.array([abs(np.sum(val_mega_histogram[:, h], dtype=np.int32)) for h in range(num_of_clusters)])
plt.bar(np.arange(num_of_clusters), vy_scalar)
plt.xlabel("Visual Word Index")
plt.ylabel("Frequency")
plt.title("Complete validation Vocabulary Generated")
# plt.xticks(np.arange(num_of_clusters) + 0.4, np.arange(num_of_clusters))
plt.show()

val_scaled_mega_histogram = scaler.transform(val_mega_histogram)
acc = svm.score(val_scaled_mega_histogram, test_labels)
print(f"SVM Accuracy:{acc*100:.2f}")

Lpre = log.predict(val_scaled_mega_histogram)
print(f"Logistic Accuracy:{accuracy_score(test_labels, Lpre)*100:.2f}")
