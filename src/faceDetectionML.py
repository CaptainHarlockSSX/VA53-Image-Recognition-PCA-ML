import sys
import numpy as np
import glob
import cv2
import tensorflow as tf

# Load all images from a database - Return a training set and a testing set
def loadImageDatabase():
    # Initialize two array
    trainSet = []
    testSet = []

    # Load arrays with images if the size is 2000x2000 px
    for img in glob.glob('../../Test_Set/*R.jpeg'):
        image = cv2.imread(img, 0)
        if image.shape == (2000,2000):
            testSet.append(image)

    for img in glob.glob('../../Train_Set/*R.jpeg'):
        image = cv2.imread(img, 0)
        if image.shape == (2000,2000):
            trainSet.append(image)

    # Convert arrays to numpy arrays
    trainSet = np.array(trainSet, dtype = int)
    testSet = np.array(testSet, dtype = int)

    # Scale colors from 0 - 255 to 0 - 1
    trainSet = trainSet / 255.0
    testSet = testSet / 255.0

    return (trainSet, testSet)

# Load data sets
(trainSet, testSet) = loadImageDatabase()
print(trainSet.shape)
print(testSet.shape)

print("OK")
# define a video capture object
# device = cv2.VideoCapture(0)

# while True:
    # ret, currentFrame = device.read()
    # currentFrame = cv2.flip(currentFrame, 1)
    #
    # cv2.imshow("Camera Stream", currentFrame)

    # if cv2.waitKey(10) == 27:
    #     break
#
# device.release()
# sys.exit() # to exit from all the processes
# cv2.destroyAllWindows() # destroy all windows
