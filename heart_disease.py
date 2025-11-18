!pip install numpy
!pip install pandas
!pip install matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

df = pd.read_csv("C:\\Users\\VICTUS\\Downloads\\heart.csv")

df.head()

X = df.drop("target", axis = 1)
print(X)
y = df['target']
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = RandomForestClassifier()
model.fit(X_train, y_train)
r = model.predict(X_test)
print(r)
print(accuracy_score(y_test, r))
print(confusion_matrix(y_test, r))
print(classification_report(y_test, r))

f1 = y_test
f2 = r
res = pd.crosstab(f1, f2)
res.plot(kind = 'bar', stacked = False, figsize = (6, 4), color = ['red', 'blue'])
plt.xlabel("actual value")
plt.ylabel("predicted value")
plt.title("actual value vs predicted value")
plt.legend(title = "actual vs predicted result")
plt.show
