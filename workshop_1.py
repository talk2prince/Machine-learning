import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix


train_set,valid_set,test_set = pickle.load(open('mnist.pkl','rb'),encoding='latin1')

X_train = train_set[0]
y_train = train_set[1]

X_valid = valid_set[0]
y_valid = valid_set[1]

X_test = test_set[0]
y_test = test_set[1]

X_newtrain = np.concatenate((X_train,X_valid),axis=0)
y_newtrain = np.concatenate((y_train,y_valid),axis=0)

X_newtrain = X_newtrain/255.
X_test = X_test/255.

mlp = MLPClassifier(hidden_layer_sizes=(784),max_iter=10,solver='adam',learning_rate_init=0.1,verbose=3)
print("Hello")
print(y_newtrain)
mlp.fit(X_newtrain,y_newtrain)

pred1 = mlp.predict(X_test)

print(classification_report(y_test,pred1))
print(confusion_matrix(y_test,pred1))
