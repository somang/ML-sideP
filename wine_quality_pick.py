## will learn wine quality, and give whether its good or bad.
import pandas as pd
import numpy as np

from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats.mstats import spearmanr
from scipy.stats.stats import pearsonr

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import SGD

def main():
  df = pd.read_csv('data/winequality-red.csv')
  '''
  print(df['fixed acidity'])
  print(df['volatile acidity'])
  print(df['citric acid'])
  print(df['residual sugar'])
  print(df['chlorides'])
  print(df['free sulfur dioxide'])
  print(df['total sulfur dioxide'])
  print(df['density'])
  print(df['pH'])
  print(df['sulphates'])
  print(df['alcohol'])

  print(df['quality'])
  
  print(spearmanr(df['fixed acidity'], df['quality']))
  print(spearmanr(df['volatile acidity'], df['quality']))
  print(spearmanr(df['citric acid'], df['quality']))
  print(spearmanr(df['residual sugar'], df['quality']))
  print(spearmanr(df['chlorides'], df['quality']))
  print(spearmanr(df['free sulfur dioxide'], df['quality']))
  print(spearmanr(df['total sulfur dioxide'], df['quality']))
  print(spearmanr(df['density'], df['quality']))
  print(spearmanr(df['pH'], df['quality']))
  print(spearmanr(df['sulphates'], df['quality']))
  print(spearmanr(df['alcohol'], df['quality']))

  print(pearsonr(df['fixed acidity'], df['quality']))
  print(pearsonr(df['volatile acidity'], df['quality']))
  print(pearsonr(df['citric acid'], df['quality']))
  print(pearsonr(df['residual sugar'], df['quality']))
  print(pearsonr(df['chlorides'], df['quality']))
  print(pearsonr(df['free sulfur dioxide'], df['quality']))
  print(pearsonr(df['total sulfur dioxide'], df['quality']))
  print(pearsonr(df['density'], df['quality']))
  print(pearsonr(df['pH'], df['quality']))
  print(pearsonr(df['sulphates'], df['quality']))
  print(pearsonr(df['alcohol'], df['quality']))
  '''
  
  dt = np.column_stack(
    (df['alcohol'], df['sulphates'], 
    df['volatile acidity'], df['citric acid'], df['quality']))

  print(dt.shape)

  # split data to training (80%) and testing set (20%)
  spliter  = int(len(df.index)*0.4)
  sc = StandardScaler()
  #tr, ts = df.values[:spliter,:], df.values[spliter:,:]
  tr, ts = dt[:spliter,:], dt[spliter:,:]
  tr_x, ts_x = sc.fit_transform(tr[:,:-1]), sc.fit_transform(ts[:,:-1]), 
  tr_y, ts_y = tr[:,-1:], ts[:,-1:]

  # What about categorical i.e. if quality is above 6.5, then it's good wine.
  # 1 = good, 0 = not good
  cat = []
  for i in df.values[:,-1:]:
    if i > 5:
      cat.append([1])
    else:
      cat.append([0])
  cat = np.asarray(cat)
  encoder = LabelEncoder()
  encoder.fit(cat)
  encoded_y = encoder.transform(cat)
  #cat = to_categorical(cat)
  #print(cat)

  tr_cat_y, ts_cat_y = cat[:spliter,:], cat[spliter:,:]
  print(tr_cat_y.shape, ts_cat_y.shape)

  # create model using sequential-categorical
  model = Sequential()
  #model.add(Dense(units=12, input_dim=11, kernel_initializer='normal', activation='relu'))
  model.add(Dense(units=12, input_dim=4, kernel_initializer='normal', activation='relu'))
  model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
  model.add(Dense(units=1, activation='sigmoid'))
  # compile
  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.7, nesterov=True)

  model.compile(
                #loss='categorical_crossentropy',
                loss='binary_crossentropy',
                optimizer=sgd, 
                metrics=['accuracy'])
  # fit the empirical value model
  hist = model.fit(tr_x, tr_cat_y, 
                epochs=600, batch_size=20,
                verbose=2, validation_data=(ts_x, ts_cat_y)
                )
  


  '''
  # create model using sequential
  model = Sequential()
  model.add(Dense(units=13, input_dim=11, activation='relu'))
  model.add(Dense(units=6, activation='relu'))
  model.add(Dense(units=1))
  # compile
  model.compile(
                loss='mean_squared_error',
                optimizer='adam', 
                metrics=['accuracy'])
  # fit the empirical value model
  hist = model.fit(tr_x, tr_y, 
                epochs=50, batch_size=12,
                verbose=2, validation_data=(ts_x, ts_y)
                )
  # make prediction
  predictions = model.predict(ts_x)
  # compare with the expected data
  rms = sqrt(mean_squared_error(ts_y, predictions))
  print("RMSE: {:.2f}".format(rms))
  correct = 0
  rounded_p = np.rint(predictions)
  for j in range(len(rounded_p)):
    if rounded_p[j] == ts_y[j]:
      correct += 1
  print("acc: {:.2f}%".format(correct/len(rounded_p)*100.0))
  '''


if __name__ == "__main__":
  main()