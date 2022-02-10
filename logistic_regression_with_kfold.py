import pandas
from sklearn import linear_model

from sklearn.model_selection import KFold

from sklearn import metrics




dataset = pandas.read_csv("dataset.csv")

target = dataset.iloc[:,2].values
data = dataset.iloc[:,3:9].values

kfold_object = KFold(n_splits=4)
kfold_object.get_n_splits(data)

print(kfold_object)


i=0
for training_index, test_index in kfold_object.split(data):
	i=i+1
	print("Round: ", str(i))
	# print("Training Index: ")
	# print(training_index)
	# print("Test Index: ")
	# print(test_index)
	# print("\n\n")
	data_training = data[training_index]
	target_training = target[training_index]
	data_test = data[test_index]
	target_test = target[test_index]
	machine = linear_model.LogisticRegression()
	machine.fit(data_training, target_training)	
	new_target = machine.predict(data_test)
	print("Accuracy Score: ",  metrics.accuracy_score(target_test, new_target))
	print("Confusion Matrix: \n", metrics.confusion_matrix(target_test, new_target))

