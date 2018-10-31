import numpy
from sklearn import datasets,metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.cross_validation import train_test_split

(X,y) = datasets.load_iris(return_X_y=True)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)

clf = MLPClassifier(solver='adam',activation='relu',hidden_layer_sizes=(65),max_iter=500)
clf.fit(X_train,y_train)

pred1 = clf.predict(X_test)

print(classification_report(y_test,pred1))
print(confusion_matrix(y_test,pred1))
