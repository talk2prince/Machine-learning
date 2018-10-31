import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix


df = pd.read_csv('fashion-mnist_train.csv')

X_train = df.drop(['label'],axis=1)
y_train = df['label']

df_test = df = pd.read_csv('fashion-mnist_test.csv')

X_test = df.drop(['label'],axis=1)
y_test = df['label']


'''
mlp = MLPClassifier(hidden_layer_sizes=(784),max_iter=25,solver='adam',learning_rate_init=0.01,verbose=3)

mlp.fit(X_train,y_train)

pred1 = mlp.predict(X_test)

print(classification_report(y_test,pred1))
print(confusion_matrix(y_test,pred1))
'''

from sklearn.grid_search import GridSearchCV
param_grid = {'hidden_layer_sizes':[(784,)],'solver':['adam','sgd','lbfgs'],'max_iter':[20,10],'learning_rate' : ['constant', 'invscaling', 'adaptive'],'learning_rate_init':[0.1,0.3,0.6],'activation' :['identity', 'logistic', 'tanh']}

grid = GridSearchCV(MLPClassifier(),param_grid,verbose=3)
grid.fit(X_train,y_train)
print(grid.best_params_)
print(grid.best_estimator_)
gridpred = grid.predict(X_test)

print(confusion_matrix(y_test,gridpred))
print(classification_report(y_test,gridpred))
