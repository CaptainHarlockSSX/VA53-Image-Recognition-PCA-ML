import sys  # to access the system
import numpy as np
import glob
import cv2

# Pick images from dataset and load them


def loadImageDatabase():
    faces = []

    for img in glob.glob('../Dataset_VA53/*R.jpeg'):
        faces.append(cv2.imread(img, 0))

    return faces


def picturesToLines(faces):
    # M pixels (width*height = M) times N pics
    pics = []
    for face in faces:
        # Each pic become one line of the MxN matrix
        pics.append(face.flatten())
    pics = np.array(pics).transpose()
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


faces = loadImageDatabase()
pics = picturesToLines(faces)
test = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8,
                9, 10, 11, 12], [7, 8, 9, 10, 11, 12, 13, 14, 15]]).transpose()
print(test)
test2 = findEigenvectors(test)
print(test2)

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
