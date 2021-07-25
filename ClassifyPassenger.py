__author__ = "Avery Chan and Sudeep Reddy"
import numpy as np
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

print("Classification starting...")

train_data = pd.read_csv("./titanic/train.csv")
# train_data_no_na = pd.read_csv("./titanic/train_no_na.csv")
test_data = pd.read_csv("./titanic/test.csv")
# test_data_no_na = pd.read_csv("./titanic/test_no_na.csv")
# print(train_data_no_na)
# print(test_data_no_na)
# print(train_data)
# print(test_data)

y = train_data["Survived"]
features = [
    # "PassengerId",
    # "Survived",
    "Pclass",
    # "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    # "Ticket",
    "Fare", # this has a single NA
    # "Cabin",
    "Embarked"
]

# Cleans data really nicely - seperates into numbers where applicable
X_train = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
print(X_train)
# print(X_test)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X_train)

X_train = imp.transform(X_train)
X_test = imp.transform(X_test)

print(f"IMP:\n{pd.DataFrame(X_train)}")

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(f"SCALE:\n{pd.DataFrame(X_train)}")

# model = RandomForestClassifier(n_estimators=10000, max_depth=15, random_state=1)
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 10), random_state=1, max_iter=100)
model.fit(X_train, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)

print("Classification end.")
