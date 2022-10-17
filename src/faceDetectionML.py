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

    # Convert arrays to numpy arrays
    testSet = np.array(testSet, dtype=np.uint8)
    testLabels = np.asarray(testLabels)

    # Scale colors from 0 - 255 to 0 - 1
    testSet = testSet / 255.0

    return (testSet, testLabels)

##################### main #####################


# Load data sets
(testSet, testLabels) = loadImageDatabase()

# Get saved ML model
model = tf.keras.models.load_model('../../SavedModels/faceRecognitionModel')

# Build the probability model to process predictions on chosen images
probabilityModel = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# Gather the predictions values
predictions = probabilityModel.predict(testSet)

for i in range(len(testSet)):
    bestPrediction = np.argmax(predictions[i])
    guess = labelID[bestPrediction]
    img = testSet[i]
    cv2.putText(img=img, text='I think you are ' + guess, org=(400, 1600),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(255, 0, 255), thickness=3)
    cv2.putText(img=img, text='Press space to continue...', org=(
        800, 1800), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 255), thickness=3)
    cv2.imshow("Heads", img)

    # Press spacebar to continue
    while(True):
        if cv2.waitKey(0) == 32:
            break

print("End Program")

sys.exit()  # Exit from all the processes
cv2.destroyAllWindows()  # Destroy all windows
