import sys
import numpy as np
import re
import glob
import cv2
import tensorflow as tf

# Table associating a number to a name
labelID = []

# Regex to get the name of the person on the image, based on the filename
def getLabel(fileName):
    pattern = re.compile('^(.*\/)*(?P<name>.+)(_[0-9]*\.jpeg){1}$')
    found = pattern.match(fileName)
    if found:
        return found.groupdict()['name']
    else:
        return ""


# Load all images from a database - Return a training set and a testing set
def loadImageDatabase():
    # Initialize two array
    trainSet, trainLabels = [], []
    testSet, testLabels = [], []

    # Load arrays with images if the size is 2000x2000 px
    for img in glob.glob('../../Test_Set/*.jpeg'):
        image = cv2.imread(img, 0)
        if image.shape == (2000, 2000):
            testSet.append(image)
            # Extract name of the person from the filename
            name = getLabel(img)

            if name not in labelID:
                labelID.append(name)

            testLabels.append(labelID.index(name))

    for img in glob.glob('../../Train_Set/*.jpeg'):
        image = cv2.imread(img, 0)
        if image.shape == (2000, 2000):
            trainSet.append(image)
            # Extract name of the person from the filename
            name = getLabel(img)

            if name not in labelID:
                labelID.append(name)

            trainLabels.append(labelID.index(name))

    # Convert arrays to numpy arrays
    trainSet = np.array(trainSet, dtype=np.uint8)
    trainLabels = np.asarray(trainLabels)
    testSet = np.array(testSet, dtype=np.uint8)
    testLabels = np.asarray(testLabels)

    # Scale colors from 0 - 255 to 0 - 1
    trainSet = trainSet / 255.0
    testSet = testSet / 255.0

    return (trainSet, trainLabels, testSet, testLabels)

@profile
def buildModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(2000, 2000)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(11)])
    return model

@profile
def compileModel(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

@profile
def trainModel(model, trainSet, trainLabels, iterations):
    model.fit(trainSet, trainLabels, epochs=iterations)
    return model

##################### main #####################


# Load data sets
(trainSet, trainLabels, testSet, testLabels) = loadImageDatabase()

# Build ML model
model = buildModel()

# Compile model
model = compileModel(model)

# Train model
model = trainModel(model, trainSet, trainLabels, 50)

# Save model
# model.save('../../SavedModels/faceRecognitionModel_50')
