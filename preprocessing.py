import pandas as pd
import numpy as np
df = pd.read_csv("income_prediction/adult (2).csv")
print(df.head())

df.replace("?", np.nan, inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)
print(df.isnull().sum())

df.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 
            'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'],
           ['divorced', 'married', 'married', 'married', 
            'not married', 'not married', 'not married'], inplace=True)
print(df['marital-status'].value_counts())

from sklearn import preprocessing

category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                'relationship', 'gender', 'native-country', 'income']

label_encoder = preprocessing.LabelEncoder()

mapping_dict = {}

for col in category_col:
    df[col] = label_encoder.fit_transform(df[col])
    mapping_dict[col] = dict(enumerate(label_encoder.classes_))

print(mapping_dict)

df.drop(['fnlwgt', 'educational-num'], axis=1, inplace=True)

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

print("Feature matrix (X):")
print(X)
print("Target vector (Y):")
print(Y)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Train model
dt_clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train, y_train)

import pickle

with open("model.pkl", "wb") as model_file:
    pickle.dump(dt_clf_gini, model_file)