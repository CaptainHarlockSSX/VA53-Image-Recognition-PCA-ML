import os
import re
import sys  # to access the system
import numpy as np
import glob
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math

# Pick images from dataset and load them


def loadImageDatabase():
    faces = []
    names = []

    for img in glob.glob('../Dataset_VA53/Train_Set/*.jpeg'):
        faces.append(cv2.imread(img, 0))
        names.append(re.findall(
            '../Dataset_VA53/Train_Set/(.*)_.*.jpeg', img)[0])

    names = np.array(names)
    faceshape = faces[0].shape
    return faces, faceshape, names


def picturesToLines(faces):
    # M pixels (width*height = M) times N samples
    A = []
    for face in faces:
        # Each sample become one line of the MxN matrix
        A.append(face.flatten())
    A = np.array(A)
    return A

# Was supposed to be the base of our own PCA method


def findEigenvectors(mat):
    # Substracting the mean vector to all vectors
    subMatrix = mat - mat.mean(axis=0, keepdims=True)

    # Pseudo-covariance matrix : transposed C dot C (CtC)
    covarMatrix = subMatrix.transpose()@subMatrix
    # w : eigenvalues of CtC, v : eigenvectors of CtC
    w, v = np.linalg.eig(covarMatrix)
    # Eigenvectors of CCt : C dot v
    eigenV = subMatrix@v
    normEigenV = eigenV/np.linalg.norm(eigenV)

    return normEigenV


def createPCAFaces(A, order):
    # matrix = findEigenvectors(A)

    pca = PCA().fit(A)

    # Choose n components for the analysis
    n_components = order
    e_faces = pca.components_[:n_components]

    return e_faces, pca


def getImageInputByName():
    while True:
        file = '../Dataset_VA53/Test_Set/' + \
            input("Person name : ") + '_0.jpeg'
        if file == "QUIT":
            break
        if os.path.exists(file):
            query = cv2.imread(file, 0).reshape(1, -1)
            break
        else:
            print("Person not found !")
    return query


def identifyFace():
    faces, faceshape, names = loadImageDatabase()
    A = picturesToLines(faces)

    e_faces, pca = createPCAFaces(A, 20)

    weights = e_faces @ (A - pca.mean_).transpose()

    # query = getImageInputByName() #if you want to test one image at a time
    test_images = glob.glob('../Dataset_VA53/Test_Set/*.jpeg')
    sample_size = len(test_images)
    rows = math.ceil(sample_size / 2)
    fig, axs = plt.subplots(2, rows)

    for idx, file in enumerate(test_images):
        query = cv2.imread(file, 0).reshape(1, -1)
        q_weight = e_faces @ (query - pca.mean_).T

        euclidean_distance = np.linalg.norm(weights - q_weight, axis=0)
        best_match = np.argmin(euclidean_distance)

        row = 0 if (idx < rows) else 1

        realname = re.findall(
            '../Dataset_VA53/Test_Set/(.*)_.*.jpeg', file)[0]

        axs[row, idx % rows].imshow(query.reshape(faceshape), cmap="gray")
        axs[row, idx % rows].set_title(
            names[best_match] + '(' + realname + ')')
        axs[row, idx % rows].set_yticklabels([])
        axs[row, idx % rows].set_xticklabels([])
    if (sample_size % 2 == 1):
        axs[-1, -1].axis('off')
    plt.show()
    return 1


identifyFace()
# loadImageDatabase()


# define a video capture object
# device = cv2.VideoCapture(0)

# while True:
#     # ret, currentFrame = device.read()
#     # currentFrame = cv2.flip(currentFrame, 1)
#     #
#     # cv2.imshow("Camera Stream", currentFrame)

#     if cv2.waitKey(10) == 27:
#         break
#
# device.release()
# sys.exit() # to exit from all the processes
# cv2.destroyAllWindows() # destroy all windows
