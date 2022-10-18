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
    trainSet, trainLabels = [], []
    testSet, testLabels = [], []

    for train in glob.glob('../../Train_Set/*.jpeg'):
        trainSet.append(cv2.imread(train, 0))
        trainLabels.append(re.findall(
            '../../Train_Set/(.*)_.*.jpeg', train)[0])

    for test in glob.glob('../../Test_Set/*.jpeg'):
        testSet.append(cv2.imread(test, 0))
        testLabels.append(re.findall(
            '../../Test_Set/(.*)_.*.jpeg', test)[0])

    trainLabels = np.array(trainLabels)
    testLabels = np.array(testLabels)
    faceshape = trainSet[0].shape
    return (trainSet, trainLabels, testSet, testLabels, faceshape)


@profile
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


@profile
def createPCAFaces(A, order):
    # matrix = findEigenvectors(A)

    pca = PCA().fit(A)

    # Choose n components for the analysis
    n_components = order
    e_faces = pca.components_[:n_components]

    return e_faces, pca


def getImageInputByName():
    while True:
        file = '../../Test_Set/' + \
            input("Person name : ") + '_0.jpeg'
        if file == "QUIT":
            break
        if os.path.exists(file):
            query = cv2.imread(file, 0).reshape(1, -1)
            break
        else:
            print("Person not found !")
    return query


@profile
def identifyFace(testSet, weights, pca, e_faces):

    # query = getImageInputByName() #if you want to test one image at a time

    best_matches = []

    for query in testSet:
        query = query.reshape(1, -1)
        q_weight = e_faces @ (query - pca.mean_).T

        euclidean_distance = np.linalg.norm(weights - q_weight, axis=0)
        best_matches.append(np.argmin(euclidean_distance))

    return best_matches


def showResults(trainLabels, testSet, testLabels, best_matches):
    sample_size = len(testSet)
    cols = math.ceil(sample_size / 2)
    fig, axs = plt.subplots(2, cols)
    for idx, best_match in enumerate(best_matches):

        # Showing results
        row = 0 if (idx < cols) else 1

        axs[row, idx % cols].imshow(
            testSet[idx].reshape(faceshape), cmap="gray")
        axs[row, idx % cols].set_title(
            trainLabels[best_match] + '(' + testLabels[idx] + ')')
        axs[row, idx % cols].set_yticklabels([])
        axs[row, idx % cols].set_xticklabels([])

    # Odd number of test images leave an empty set of axis in the last subplot
    if (sample_size % 2 == 1):
        axs[-1, -1].axis('off')
    plt.show()

    return 1


##################### main #####################
(trainSet, trainLabels, testSet, testLabels, faceshape) = loadImageDatabase()
A = picturesToLines(trainSet)

e_faces, pca = createPCAFaces(A, 1)

weights = e_faces @ (A - pca.mean_).transpose()
best_matches = identifyFace(testSet, weights, pca, e_faces)
showResults(trainLabels, testSet, testLabels, best_matches)
