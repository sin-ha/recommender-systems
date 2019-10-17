# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:37:31 2019

@author: HARSH
"""
#from sklearn.utils import shuffle
import pickle
with open('neighbours.json', 'rb') as f:
  neighbors = pickle.load(f)

with open('deviations.json', 'rb') as f:
  deviations = pickle.load(f)

with open('um2r.json', 'rb') as f:
  usermovie2rating = pickle.load(f)
with open('averages.json', 'rb') as f:
  averages = pickle.load(f)
#<---------------------------------------------for testing--------------------------->
"""with open('usermovie2rating_test.json', 'rb') as f:
  usermovie2rating_test = pickle.load(f)
usermovie2rating = shuffle(usermovie2rating)
cutoff = int(0.8*len(usermovie2rating))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]
"""
def predict(i, m):
  # calculate the weighted sum of deviations
  numerator = 0
  denominator = 0
  for neg_w, j in neighbors[i]:
    # remember, the weight is stored as its negative
    # so the negative of the negative weight is the positive weight
    try:
      numerator += -neg_w * deviations[j][m]
      denominator += abs(neg_w)
    except KeyError:
      # neighbor may not have rated the same movie
      # don't want to do dictionary lookup twice
      # so just throw exception
      pass

  if denominator == 0:
    prediction = averages[i]
  else:
    prediction = numerator / denominator + averages[i]
  prediction = min(5, prediction)
  prediction = max(0.5, prediction) # min rating is 0.5
  return prediction


train_predictions = []
train_targets = []
for (i, m), target in usermovie2rating.items():
  # calculate the prediction for this movie
  prediction = predict(i, m)

  # save the prediction and target
  train_predictions.append(prediction)
  train_targets.append(target)

"""test_predictions = []
test_targets = []
# same thing for test set
for (i, m), target in usermovie2rating_test.items():
  # calculate the prediction for this movie
  prediction = predict(i, m)

  # save the prediction and target
  test_predictions.append(prediction)
  test_targets.append(target)
"""
import numpy as np
# calculate accuracy
def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean(abs(p - t))

print('train mse:', mse(train_predictions, train_targets))
#print('test mse:', mse(test_predictions, test_targets))

