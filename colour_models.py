from sklearn.preprocessing import MinMaxScaler,LabelEncoder,PolynomialFeatures,FunctionTransformer
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from skimage.color import rgb2lab, rgb2hsv

def rgb_to_lab(X):
    """
    Convert a 2D array of RGB colour values to LAB values.
    
    Reshaping is necessary because the skimage functions expect an image: width * height * 3 array.
    We have/need a n * 3 array and must convert there and back.
    """
    return rgb2lab(X.reshape(1,-1,3)).reshape(-1,3)


def rgb_to_hsv(X):
    """
    Convert a 2D array of RGB colour values to HSV values.
    
    Reshaping is necessary because the skimage functions expect an image: width * height * 3 array.
    We have/need a n * 3 array and must convert there and back.
    """
    return rgb2hsv(X.reshape(1,-1,3)).reshape(-1,3)


# This is a function to use to convert rgb to other color channel. Set to None using rgb.
colorFunc = rgb_to_lab

# Naive Bayes Classifier
def nbModel():
    return make_pipeline(
        FunctionTransformer(func=colorFunc, validate=False),
        GaussianNB()
    )

# Gradient Boosting Classifier
def gbModel():
    return make_pipeline(
        FunctionTransformer(func=colorFunc, validate=False),
        GradientBoostingClassifier(n_estimators=30,
        max_depth=3, min_samples_leaf=0.1)
    )

# k-Nearest Neighbors Classifier
def knnModel():
    return make_pipeline(
        FunctionTransformer(func=colorFunc, validate=False),
        KNeighborsClassifier(n_neighbors=8)
    )

# Random Forest Classifier
def rfModel():
    return make_pipeline(
        FunctionTransformer(func=colorFunc, validate=False),
        RandomForestClassifier(n_estimators=380, max_depth=8, min_samples_leaf=8)
    )

# SVM Classifier
def svmModel():
    return make_pipeline(
        FunctionTransformer(func=colorFunc, validate=False),
        MinMaxScaler(),
        SVC(kernel='rbf', C=650, gamma=3),
    )

# Neural Network Classifier
def nnModel():
    return make_pipeline(
        FunctionTransformer(func=colorFunc, validate=False),
        MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25,), random_state=1)
    )