import tensorflow
import keras
import sklearn
from sklearn import datasets
from sklearn import svm

# print("SVM is real")

# loading breast cancer dataset
cancer_data = datasets.load_breast_cancer()

print(cancer_data.feature_names, "\n")
print(cancer_data.target_names, "\n")

X = cancer_data.data
Y = cancer_data.target
# split data into train data and test data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

# print(x_train,"\n", y_train)

target_classes=['malignant','benign']

