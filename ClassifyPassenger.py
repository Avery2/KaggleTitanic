__author__ = "Avery Chan and Sudeep Reddy"
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

print("Classification starting...")

train_data = pd.read_csv("./titanic/train.csv")
test_data = pd.read_csv("./titanic/test.csv")
print(train_data)
print(test_data)

y = train_data["Survived"]
features = [
    # "PassengerId",
    # "Survived",
    "Pclass",
    # "Name",
    "Sex",
    # "Age",
    "SibSp",
    "Parch",
    # "Ticket",
    # "Fare", # this has a single NA
    # "Cabin",
    # "Embarked"
]

# Cleans data really nicely - seperates into numbers where applicable
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
print(X)
print(X_test)

model = RandomForestClassifier(n_estimators=1000, max_depth=8, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)

print("Classification end.")
