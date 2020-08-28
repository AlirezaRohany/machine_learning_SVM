import tensorflow
import keras
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# print("SVM is real")

# loading breast cancer dataset
breast_cancer_data = datasets.load_breast_cancer()

print("Features: \n", breast_cancer_data.feature_names, "\n")
print("Target classes: \n", breast_cancer_data.target_names, "\n")

X = breast_cancer_data.data
Y = breast_cancer_data.target

# split data into train data and test data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# print(x_train,"\n", y_train)

target_classes = ['malignant', 'benign']

# making svm classifier and training data
svm_classifier = svm.SVC(kernel='linear', C=2)
svm_classifier.fit(x_train, y_train)

# print("\n",svm_classifier.kernel,svm_classifier.degree,"\n")

# predict the class of test data
svm_predictions = svm_classifier.predict(x_test)

svm_accuracy = metrics.accuracy_score(y_test, svm_predictions)
print("SVM accuracy:", svm_accuracy, "\n")

# comparing to knn classifier
knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(x_train, y_train)
knn_predictions = knn_classifier.predict(x_test)
knn_accuracy = metrics.accuracy_score(y_test, knn_predictions)
print("KNN accuracy:", knn_accuracy)
