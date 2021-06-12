import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf

from sklearn.model_selection import train_test_split

animals_dataset_directory_path = 'animals_dataset'
animals_categories = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
animals_dataset = []

for single_animals_category in animals_categories:
    single_animals_category_directory_path = os.path.join(animals_dataset_directory_path, single_animals_category)
    single_animals_category_label = animals_categories.index(single_animals_category)
    for single_animal_image in os.listdir(single_animals_category_directory_path):
        single_animal_image_path = os.path.join(single_animals_category_directory_path, single_animal_image)
        animal_image = cv.imread(single_animal_image_path)
        animal_image = cv.resize(animal_image, (128, 128))
        animals_dataset.append([animal_image, single_animals_category_label])

random.shuffle(animals_dataset)

animal_image_array = []
animal_label_array = []
for animal_image, animal_label in animals_dataset:
    animal_image_array.append(animal_image)
    animal_label_array.append(animal_label)

animal_image_array = np.array(animal_image_array)
animal_label_array = np.array(animal_label_array)

animal_image_training, animal_image_testing, animal_label_training, animal_label_testing = train_test_split(
    animal_image_array, animal_label_array, test_size=0.20, random_state=42)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16,
                                 kernel_size=2,
                                 padding="same",
                                 activation=tf.keras.activations.relu,
                                 input_shape=(128, 128, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding="same", activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(500, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))

# Define the Gradient Descend, the Loss Function and the metrics.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model.
model.fit(animal_image_training, animal_label_training, batch_size=50, epochs=100, verbose=1)

# Calculate the accuracy and the loss.
loss, accuracy = model.evaluate(animal_image_testing, animal_label_testing)
print(f"The model's accuracy is '{accuracy}'.")
print(f"The model's loss is '{loss}'.")

model.save('model')

for single_image_index in range(0, 16):
    single_image = cv.imread(f'./animals/{single_image_index}.png')
    single_image = cv.resize(single_image, (128, 128))
    single_image = np.expand_dims(single_image, axis=0)

    single_image_prediction_index = model.predict(single_image)
    single_image_prediction_index = np.argmax(single_image_prediction_index)

    animal_prediction = ''
    if single_image_prediction_index == 0:
        animal_prediction = 'butterfly'
    if single_image_prediction_index == 1:
        animal_prediction = 'cat'
    if single_image_prediction_index == 2:
        animal_prediction = 'chicken'
    if single_image_prediction_index == 3:
        animal_prediction = 'cow'
    if single_image_prediction_index == 4:
        animal_prediction = 'dog'
    if single_image_prediction_index == 5:
        animal_prediction = 'elephant'
    if single_image_prediction_index == 6:
        animal_prediction = 'horse'
    if single_image_prediction_index == 7:
        animal_prediction = 'sheep'
    if single_image_prediction_index == 8:
        animal_prediction = 'spider'
    if single_image_prediction_index == 9:
        animal_prediction = 'squirrel'

    print(f"For the image number '{single_image_index}', the predicted result is probably a '{animal_prediction}'.")
    plt.imshow(single_image[0], cmap=plt.cm.binary)  # The 'cmap=plt.cm.binary' code draws a black and white picture.
    plt.show()
