import pandas
from sklearn import linear_model

dataset = pandas.read_csv("dataset.csv")
# print(dataset)

target = dataset.iloc[:,0].values
# print(target)

data = dataset.iloc[:,3:9].values
# print(data)


machine = linear_model.LinearRegression()

machine.fit(data, target)


new_data = [
	[-0.5,1.1,0.88,0.4,3,0],
	[0.6,1.4,-0.1,1,2,1]
]

new_target = machine.predict(new_data)
print(new_target)