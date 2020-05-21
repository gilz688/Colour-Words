import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from skimage.color import lab2rgb, hsv2rgb
from colour_models import nbModel, gbModel, knnModel, rfModel, svmModel, nnModel

# representative RGB colours for each label, for nice display
COLOUR_RGB = {
    'red': (255, 0, 0),
    'orange': (255, 114, 0),
    'yellow': (255, 255, 0),
    'green': (0, 230, 0),
    'blue': (0, 0, 255),
    'purple': (187, 0, 187),
    'brown': (117, 60, 0),
    'pink': (255, 187, 187),
    'black': (0, 0, 0),
    'grey': (150, 150, 150),
    'white': (255, 255, 255),
}
name_to_rgb = np.vectorize(COLOUR_RGB.get, otypes=[np.uint8, np.uint8, np.uint8])


def plot_predictions_lab(model, lum=70, resolution=256):
    """
    Create a slice of LAB colour space with given luminance; predict with the model; plot the results.
    """
    wid = resolution
    hei = resolution
    n_ticks = 5

    # create a hei*wid grid of LAB colour values, with L=lum
    ag = np.linspace(-100, 100, wid)
    bg = np.linspace(-100, 100, hei)
    aa, bb = np.meshgrid(ag, bg)
    ll = lum * np.ones((hei, wid))
    lab_grid = np.stack([ll, aa, bb], axis=2)

    # convert to RGB for consistency with original input
    X_grid = lab2rgb(lab_grid)

    # predict and convert predictions to colours so we can see what's happening
    y_grid = model.predict(X_grid.reshape((wid*hei, 3)))
    pixels = np.stack(name_to_rgb(y_grid), axis=1) / 255
    pixels = pixels.reshape((hei, wid, 3))

    # plot input and predictions
    plt.figure(figsize=(10, 5))
    plt.suptitle('Predictions at L=%g' % (lum,))
    plt.subplot(1, 2, 1)
    plt.title('Inputs')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.ylabel('B')
    plt.imshow(X_grid.reshape((hei, wid, 3)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.imshow(pixels)


def plot_predictions_hsv(model, val=0.8, resolution=256):
    """
    Create a slice of HSV colour space with given luminance; predict with the model; plot the results.
    """
    wid = resolution
    hei = resolution
    n_ticks = 5

    # create a hei*wid grid of HSV colour values, with V=Val
    hg = np.linspace(0.0, 1.0, wid)
    sg = np.linspace(0.0, 1.0, hei)
    hh, ss = np.meshgrid(hg, sg)
    vv = val * np.ones((hei, wid))
    hsv_grid = np.stack([hh, ss, vv], axis=2)

    # convert to RGB for consistency with original input
    X_grid = hsv2rgb(hsv_grid)

    # predict and convert predictions to colours so we can see what's happening
    y_grid = model.predict(X_grid.reshape((wid*hei, 3)))
    pixels = np.stack(name_to_rgb(y_grid), axis=1) / 255
    pixels = pixels.reshape((hei, wid, 3))

    # plot input and predictions
    plt.figure(figsize=(10, 5))
    plt.suptitle('Predictions at V=%g' % (val,))
    plt.subplot(1, 2, 1)
    plt.title('Inputs')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(0, 1, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(0, 1, n_ticks))
    plt.xlabel('H')
    plt.ylabel('S')
    plt.imshow(X_grid.reshape((hei, wid, 3)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(0, 1, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(0, 1, n_ticks))
    plt.xlabel('H')
    plt.imshow(pixels)

def main():
    data = pd.read_csv('colour-data.csv')
    X = data[['R', 'G', 'B']].values / 255
    y = data['Label'].values

    print(X[0])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=8)

    # create and train models; have a look at some scores.
    classifiers = [
        { 'name': 'Naive Bayes', 'abrev': 'nb', 'model': nbModel() },
        { 'name': 'Gradient Boosting', 'abrev': 'gb', 'model': gbModel()},
        { 'name': 'k-Nearest Neighbors', 'abrev': 'knn', 'model': knnModel()},
        { 'name': 'Random Forest', 'abrev': 'rf', 'model': rfModel()},
        { 'name': 'Support Vector Machine', 'abrev': 'svm', 'model': svmModel()},
        { 'name': 'Neural Network', 'abrev': 'nn', 'model': nnModel()}
    ]

    for classifier in classifiers:
        print(classifier['name'])
        model = classifier['model']
        model.fit(X_train, y_train)
        print("Training Data   : ", model.score(X_train, y_train))
        print("Validation Data : ", model.score(X_valid, y_valid))
        print()

        # These lines will produce nice visualizations of your predictions, once you have some.
        plot_predictions_lab(model)
        plt.savefig('predictions-lab-' + classifier['abrev'] + '.png')
        plot_predictions_hsv(model)
        plt.savefig('predictions-hsv-' + classifier['abrev'] + '.png')

if __name__ == '__main__':
    main()
