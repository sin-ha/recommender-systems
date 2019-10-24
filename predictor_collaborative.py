# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:37:31 2019

@author: HARSH
"""
#from sklearn.utils import shuffle
import pickle
#with open('neighbours.json', 'rb') as f:
#  neighbors = pickle.load(f)
#for improvement

with open('neighbors.json', 'rb') as f:
  neighbors = pickle.load(f)
with open('deviations.json', 'rb') as f:
  deviations = pickle.load(f)

with open('um2r.json', 'rb') as f:
  usermovie2rating = pickle.load(f)
with open('averages.json', 'rb') as f:
  averages = pickle.load(f)
  
with open('df_test.json', 'rb') as f:
  df_test = pickle.load(f)
#<---------------------------------------------for testing--------------------------->
with open('usermovie2rating_test.json', 'rb') as f:
  usermovie2rating_test = pickle.load(f)

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

test_predictions = []
test_targets = []
# same thing for test set
for (i, m), target in usermovie2rating_test.items():
  # calculate the prediction for this movie
  prediction = predict(i, m)

  # save the prediction and target
  test_predictions.append(prediction)
  test_targets.append(target)

import numpy as np
# calculate accuracy
def mae(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean(abs(p - t))

print('train mae:', mae(train_predictions, train_targets))
print('test mae:', mae(test_predictions, test_targets))



#<---------------visuals------------->
import matplotlib.pyplot as plt
def visuals(k):
    s = []
    for i in range(int(len(k)/1000)):
        m = np.mean(k[i*1000:(i+1)*1000])+0.1
        s.append(m)
    return s
plt.plot(visuals(train_predictions),label = "train_predictions")
plt.plot(visuals(train_targets),label = "train_targets")
plt.legend()
plt.show()

plt.plot(visuals(test_predictions),label = "test_predictions")
plt.plot(visuals(test_targets),label = "test_targets")
plt.legend()
plt.show()


plt.plot(abs(np.array(train_predictions)-np.array(train_targets)))
plt.plot(abs(np.array(test_predictions)-np.array(test_targets)))

