__author__ = "Avery Chan and Sudeep Reddy"
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

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

LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=1000)
LR.fit(X_train_imp, y)
predictions = LR.predict(X_test_imp)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)

print("Classification end.")

