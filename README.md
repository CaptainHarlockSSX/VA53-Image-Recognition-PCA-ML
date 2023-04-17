# VA53 - Benchmarking PCA and Machine Learning algorithms for image recognition purpose

## Descritption

University practical work aiming to compare **PCA** and **Machine Learning** methods to process image recognition on a small set of images. 

In our case, it was a **Face Recognition** using a <u>private dataset</u> of people faces with names.

Is has been done for the Course Credits **VA53 - Probabilistic and stochastic models for computer vision** at <u>Université de Technologie de Belfort Montbéliard (FR).</u>

## How to use

- **Define your images paths in all the files**, in the function `loadImageDatabase`

- Edit `pca.py` line 128 to set the number of PCA components you want, then run the program with Python 3.

- Edit `modelTraining.py` line 95 to set the number of iterations you want for the ML model, then line 98 to choose where to store the model. Run the code with Python 3.

- Edit `faceRecognition.py` line 63 to set the path of a previously stored model. Run the code with Python 3.
