import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from word2number import w2n
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics


import warnings
warnings.filterwarnings(action='ignore')

# Read Data
data = pd.read_csv("Clean_Dataset.csv")
print(len(data))
'''
print(data.columns)
data = data.drop(['flight','duration'], axis=1)
data.isnull().sum()
print(data.columns)
for i in range(len(data)):
    if data['stops'][i] == 'two_or_more':
        data['stops'][i] = 2
    else:
        try:
            data['stops'][i] = w2n.word_to_num(str(data['stops'][i]))
        except ValueError as e:
            print(f"Error processing value at index {i}: {data['stops'][i]} - {e}")

# Categorical to numerical values

data.to_csv("new_data.csv")
final_data=pd.read_csv("new_data.csv")

categorical_columns = ['airline', 'source_city', 'departure_time', 'arrival_time', 'destination_city', 'class']
data_encoded = pd.get_dummies(final_data, columns=categorical_columns)
final_data.to_csv("encoded.csv")
X=data_encoded[data_encoded.columns[~data_encoded.columns.isin(['price'])]]
y=data_encoded['price']

X_train,X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
print(X)

# Decision Tree Regressor
model=DecisionTreeRegressor(max_depth=5, min_samples_split=2, min_samples_leaf=1)
model.fit(X,y)
print(model.score(X,y))

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
'''
def model_prediction(airways,source_city,destination_city,departure_time,arrival_time,class_type,stops,days_left):

    final_data=pd.read_csv("new_data.csv")
    final_data=final_data._append({
    'airline': airways,
    'source_city':source_city,
    'departure_time': departure_time,
    'stops': int(stops),
    'arrival_time': arrival_time,
    'destination_city': destination_city,
    'class': class_type,
    'days_left': int(days_left)},ignore_index=True)

    categorical_columns = ['airline', 'source_city', 'departure_time', 'arrival_time', 'destination_city', 'class']
    data_encoded = pd.get_dummies(final_data, columns=categorical_columns)
    final_data.to_csv("encoded.csv")
    X=data_encoded[data_encoded.columns[~data_encoded.columns.isin(['price'])]]
    X=X.iloc[:,1:]
    print(X)
    y=data_encoded['price']
    y=y.iloc[:-1]
    l = X.iloc[-1]
    print(l)
    print("=====================")
    loaded_model = pickle.load(open("model_linear.sav", 'rb'))
    # print(int(loaded_model.predict([l])))

    return str(int(loaded_model.predict([l])))






