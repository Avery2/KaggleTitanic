__author__ = "Avery Chan and Sudeep Reddy"
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

print("Classification starting...")

train_data = pd.read_csv("./titanic/train.csv")

# print(train_data)
# men = train_data.loc[train_data.Sex == 'male']["Survived"]

print("Classification end.")