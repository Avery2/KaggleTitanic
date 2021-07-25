__author__ = "Avery Chan and Sudeep Reddy"
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

print("Classification starting...")

train_data = pd.read_csv("./titanic/train.csv")
# train_data_no_na = pd.read_csv("./titanic/train_no_na.csv")
test_data = pd.read_csv("./titanic/test.csv")
# test_data_no_na = pd.read_csv("./titanic/test_no_na.csv")
# print(train_data_no_na)
# print(test_data_no_na)
print(train_data)
print(test_data)

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
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
print(X)
print(X_test)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X)
X_train_imp = imp.transform(X)
X_test_imp = imp.transform(X_test)

model = RandomForestClassifier(n_estimators=10000, max_depth=15, random_state=1)
model.fit(X_train_imp, y)
predictions = model.predict(X_test_imp)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)

print("Classification end.")
