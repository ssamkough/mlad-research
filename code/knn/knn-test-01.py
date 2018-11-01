import numpy as np
from sklearn import preprocessing, neighbors, model_selection
import pandas as pd

df = pd.read_csv('../../datasets/breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True) # removes the id column to not skew results
print(df.head(5))
print(df.columns)
x = np.array(df.drop(columns=['class'], axis=1)) # features - everything except the class column
y = np.array(df['class']) # labels - just the class

X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2) # quickly shuffle the data and separate it into training and testing chunks

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)


accuracy = clf.score(X_test, y_test)
print(accuracy * 100)
