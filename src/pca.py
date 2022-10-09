import sys  # to access the system
import numpy as np
import glob
import cv2

#  read and image


def loadImageDatabase():
    faces = []

    for img in glob.glob('../Dataset_VA53/*R.jpeg'):
        faces.append(cv2.imread(img, 0))

    return faces


def matrixToLine(faces):
    matrix = []
    for face in faces:
        matrix.append(face.flatten())
    return matrix


faces = loadImageDatabase()
matrix = matrixToLine(faces)

# define a video capture object
# device = cv2.VideoCapture(0)

while True:
    # ret, currentFrame = device.read()
    # currentFrame = cv2.flip(currentFrame, 1)
    #
    # cv2.imshow("Camera Stream", currentFrame)

    if cv2.waitKey(10) == 27:
        break
#
# device.release()
# sys.exit() # to exit from all the processes
# cv2.destroyAllWindows() # destroy all windows
