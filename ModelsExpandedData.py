import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import pickle

#Import Data
df = pd.read_csv('expandeddata.csv')

#Performing Label encoding for Categorical Features
print(df.columns)

labelencoder = LabelEncoder()
labelencoder2 = LabelEncoder()

dfartists = pd.DataFrame(columns = ['Artists', 'Encoded_Artists'])
dfartists['Artists'] = df['artists'].unique()
df['artists'] = labelencoder.fit_transform(df['artists'])

filehandler = open("artistexpanded.obj","wb")
pickle.dump(labelencoder,filehandler)
filehandler.close()

dfgenres = pd.DataFrame(columns = ['Genres', 'Encoded_Genres'])
dfgenres['Genres'] = df['genres'].unique()
df['genres'] = labelencoder2.fit_transform(df['genres'])

filehandler2 = open("genreexpanded.obj","wb")
pickle.dump(labelencoder2,filehandler2)
filehandler2.close()

file = open("artistexpanded.obj",'rb')
artistle = pickle.load(file)
file.close()

file = open("genreexpanded.obj",'rb')
genristle = pickle.load(file)
file.close()

dfartists['Encoded_Artists'] = artistle.transform(dfartists['Artists'])
dfgenres['Encoded_Genres'] = genristle.transform(dfgenres['Genres'])

df_model = df[['artists', 'genres', 'tempo','energy',
       'danceability', 'loudness', 'liveness', 'valence', 'duration_ms',
       'acousticness', 'speechiness', 'popularity']]

scaler = StandardScaler()
scaler.fit(df_model)

X= df_model.drop('popularity', axis=1)
y= df_model.popularity

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)

#Import Packages for Models
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Create Logistic Regression and Train:
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_predictions = lr.predict(X_test)
#Evaluate the X test set for Logisitic Regression:
lr_MAE = metrics.mean_absolute_error(y_test, lr_predictions)
print('Mean Absolute Error for Multi Linear Regression:' + str(lr_MAE))


#Create Gradient Boosting Classifier and Train:
gbc = GradientBoostingRegressor(random_state=42)
gbc.fit(X_train, y_train)
gbc_predict = gbc.predict(X_test)

#Evaluate the X test set for Gradient Boosting:
gbc_MAE = metrics.mean_absolute_error(y_test, gbc_predict)
print('Mean Absolute Error for Gradient Boosting:' + str(gbc_MAE))

#Create Decision Tree Classifier and Train:
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_predict = dt.predict(X_test)

#Evaluate the X test set for Decision Tree Classifier:
dt_MAE = metrics.mean_absolute_error(y_test, dt_predict)
print('Mean Absolute Error for Decision Tree:' + str(dt_MAE))

#Create Random Forest and Train:
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_predict = rf.predict(X_test)

#Evaluate Random Forest:
#Evaluate the X test set for Random Forest Classifier:
rf_MAE = metrics.mean_absolute_error(y_test, rf_predict)
print('Mean Absolute Error for Random Forest:' + str(rf_MAE))

#Create Light GBM Model:
hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l1'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 8,
    "num_leaves": 128,
    "max_bin": 512,
    "num_iterations": 100000,
    "n_estimators": 1000
}
gbm = lgb.LGBMRegressor(**hyper_params)
gbm.fit(X_train, y_train,eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=1000)
gbm_predict = gbm.predict(X_test)

gbm_MAE = metrics.mean_absolute_error(y_test, gbm_predict)
print('Mean Absolute Error for GBM:' + str(gbm_MAE))

test_pred = np.expm1(gbm.predict(X_test, num_iteration=gbm.best_iteration_))
print('The MAE of prediction is:', round(metrics.mean_absolute_error(test_pred, y_test),2))


#Evaluating a Keras Model:
df = pd.read_csv('expandeddata.csv')
df_model = df[['artists', 'genres', 'tempo','energy',
       'danceability', 'loudness', 'liveness', 'valence', 'duration_ms',
       'acousticness', 'speechiness', 'popularity']]

dataset = pd.get_dummies(df_model, prefix='', prefix_sep='')
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("popularity")
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop('popularity')
test_labels = test_dataset.pop('popularity')

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mae',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

kerasmodel = build_model()

EPOCHS = 20000

history = keras.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=1)

loss, mae, mse = keras.evaluate(normed_test_data, test_labels, verbose=2)