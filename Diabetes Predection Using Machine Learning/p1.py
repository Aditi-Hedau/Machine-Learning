# import lib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# load the data
data = pd.read_csv("diabetes.csv")
print(data.head())

# understand the data
print(data.isnull().sum())

# features and target
features = data[["FS", "FU"]]
target = data["Diabetes"]

# handle cat data
new_features = pd.get_dummies(features, drop_first=True)

# train and test
x_train, x_test, y_train, y_test = train_test_split(new_features, target)

# model and performance
model1 = LogisticRegression()
model1.fit(x_train, y_train)
y_pred = model1.predict(x_test)
cr = classification_report(y_test, y_pred)
print(cr)

model2 = DecisionTreeClassifier(criterion="gini")
model2.fit(x_train, y_train)
y_pred = model2.predict(x_test)
cr = classification_report(y_test, y_pred)
print(cr)

model3 = RandomForestClassifier(n_estimators=10)
model3.fit(x_train, y_train)
y_pred = model3.predict(x_test)
cr = classification_report(y_test, y_pred)
print(cr)

# save the model
with open("db.model", "wb") as f:
	pickle.dump(model3, f)