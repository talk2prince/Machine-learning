import numpy
from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier

(data, targets)=datasets.load_iris(return_X_y=True); 

trainingset=data[range(0,150,2),:]
trainingsettarget=targets[range(0,150,2)]

testset=data[range(1,150,2),:]
testsettarget=targets[range(1,150,2)]

#creating instance of the classifier
clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(65), max_iter=200)

# activation : {'identity', 'logistic', 'tanh', 'relu'}, default 'relu'

#train the model
clf.fit(trainingset, trainingsettarget);

#predict using the learnt classifier
prediction = clf.predict(testset) 

print("############### Predictions #################")
print(prediction)
print("#############################################")

print("Accuracy =",metrics.accuracy_score(testsettarget, predictions, normalize=True))
