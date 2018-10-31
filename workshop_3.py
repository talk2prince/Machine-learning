import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
import seaborn as sns

df = pd.read_csv('data.csv')



df.drop('id',axis=1,inplace=True)
print(df.head())
df.drop('Unnamed: 32',axis=1,inplace=True)

X = df.drop('diagnosis',axis=1)
y = df['diagnosis']



LabelEncoder_y = LabelEncoder()
y = LabelEncoder_y.fit_transform(y)
print(y)

sns.heatmap(X.isnull(),yticklabels=False,cbar=False)
plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

'''
mlp = MLPClassifier(hidden_layer_sizes=(31,15,5),max_iter=25,solver='adam',learning_rate_init=0.4)

mlp.fit(X_train,y_train)

pred1 = mlp.predict(X_test)

print(classification_report(y_test,pred1))
print(confusion_matrix(y_test,pred1))
'''

from sklearn.grid_search import GridSearchCV
param_grid = {'hidden_layer_sizes':[(31),(31,15),(31,15,5)],'solver':['adam','sgd'],'max_iter':[100,25,10],'learning_rate_init':[0.1,0.3,0.6]}

grid = GridSearchCV(MLPClassifier(),param_grid,verbose=3)
grid.fit(X_train,y_train)
print(grid.best_params_)
print(grid.best_estimator_)


gridpred = grid.predict(X_test)

print(confusion_matrix(y_test,gridpred))
print(classification_report(y_test,gridpred))
