import os
import sys  # to access the system
import numpy as np
import glob
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Pick images from dataset and load them


def loadImageDatabase():
    faces = []

    for img in glob.glob('../Dataset_VA53/Train_Set/*.jpeg'):
        faces.append(cv2.imread(img, 0))

    faceshape = faces[0].shape

    return faces, faceshape


def picturesToLines(faces):
    # M pixels (width*height = M) times N pics
    pics = []
    for face in faces:
        # Each pic become one line of the MxN matrix
        pics.append(face.flatten())
    pics = np.array(pics)
    return pics


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


faces, faceshape = loadImageDatabase()
pics = picturesToLines(faces)
# matrix = findEigenvectors(pics)

pca = PCA().fit(pics)

# Take the first K principal components as eigenfaces
n_components = 16
eigenpics = pca.components_[:n_components]

weights = eigenpics @ (pics - pca.mean_).transpose()

while True:
    file = '../Dataset_VA53/Test_Set/' + input("Person name : ") + '_0.jpeg'
    if file == "QUIT":
        break
    if os.path.exists(file):
        query = cv2.imread(file, 0).reshape(1, -1)
        break
    else:
        print("Person not found !")

query_weight = eigenpics @ (query - pca.mean_).T

euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
best_match = np.argmin(euclidean_distance)
# Visualize
fig, axes = plt.subplots(1, 2)
axes[0].imshow(query.reshape(faceshape), cmap="gray")
axes[0].set_title("Query")
axes[1].imshow(pics[best_match].reshape(faceshape), cmap="gray")
axes[1].set_title("Best match")
plt.show()

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
