import numpy
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import pickle

#load data from a pickle file.
#if you are unsure what is inside pkl file (typically available in readme), see the type of return value and further drill down
train_set, valid_set, test_set = pickle.load(open('mnist.pkl','rb'),encoding='latin1')

#drill down
print(type(train_set), type(valid_set), type(test_set))
print(type(train_set[0]), type(valid_set[0]), type(test_set[0]))
print(len(train_set), len(valid_set), len(test_set))
print(type(train_set[1]), type(valid_set[1]), type(test_set[1]))
print(train_set[0].shape, valid_set[0].shape, test_set[0].shape)
print(train_set[1].shape, valid_set[1].shape, test_set[1].shape)

#preparing data
trainset=train_set[0]
trainsettarget=train_set[1]
validationset=valid_set[0]
validationsettarget=valid_set[1]
testset=test_set[0]
testsettarget=test_set[1]

#merging traingset and validationset in trainingset
trainset=numpy.concatenate((trainset,validationset),axis=0)
trainsettarget=numpy.concatenate((trainsettarget,validationsettarget),axis=0)

#normalization in the range 0-1
trainset=trainset/255.
testset=testset/255.

#creating instance of the classifier
mlp = MLPClassifier(hidden_layer_sizes=(784,), max_iter=10, solver='adam', learning_rate_init=0.1, verbose=10)

#train the model
mlp.fit(trainset,trainsettarget)

#predict using the learnt classifier
prediction=mlp.predict(testset)

print("############### Predictions #################")
print(prediction)
print("#############################################")

print("Accuracy =",metrics.accuracy_score(testsettarget, prediction, normalize=True))

#print(clf.predict_proba(testset));