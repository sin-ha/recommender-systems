# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 12:24:43 2019

@author: HARSH
"""
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
#<------------------loading the data------------->
import pickle
(f,g,h) = open("um2r.json","rb"), open("user2movie.json","rb"),open("movie2user.json","rb")
user2movie = pickle.load(g)
movie2user  = pickle.load(h)
usermovie2rating = pickle.load(f)
f.close()
g.close()
h.close()
i = open("usermovie2rating_test.json","rb")
usermovie2rating_test = pickle.load(i)
i.close()

#<---------------defining sum of squared losses----------------------------->
def loss(d):
    sse =0 
    for (user,movie),rating in d.items():
         my_pred = W[user].dot(U[movie]) + b[user] + c[movie] + mu
         loss = my_pred - rating
         se = loss**2
         sse+=se
    return sse/len(d)
#<-------------------------------randomly initaialising the paraeters----------->
K = 10
N= len(user2movie)
M = len(movie2user)
W = np.random.normal(3.4283012978499814,1.0220310184294517,(N,K))
b = np.zeros(N)
U = np.random.normal(3.4283012978499814,1.0220310184294517,(M,K))
c = np.zeros(M)
mu = np.mean(list(usermovie2rating.values()))

#<---------------model----------------------->\
train_losses,test_losses =[],[]
epochs = 25
reg = 0.23
for epoch in range(epochs):
  print("epoch:", epoch)
  epoch_start = datetime.now()
  # perform updates

  # update W and b
  t0 = datetime.now()
  for i in range(N):
    # for W
    matrix = np.eye(K) * reg
    vector = np.zeros(K)

    # for b
    bi = 0
    for j in user2movie[i]:
      r = usermovie2rating[(i,j)]
      matrix += np.outer(U[j], U[j])
      vector += (r - b[i] - c[j] - mu)*U[j]
      bi += (r - W[i].dot(U[j]) - c[j] - mu)

    # set the updates
    W[i] = np.linalg.solve(matrix, vector)
    b[i] = bi / (len(user2movie[i]) + reg)

    if i % (N//10) == 0:
      print("i:", i, "N:", N)
  print("updated W and b:", datetime.now() - t0)

  # update U and c
  t0 = datetime.now()
  for j in range(M):
    # for U
    matrix = np.eye(K) * reg
    vector = np.zeros(K)

    # for c
    cj = 0
    try:
      for i in movie2user[j]:
        r = usermovie2rating[(i,j)]
        matrix += np.outer(W[i], W[i])
        vector += (r - b[i] - c[j] - mu)*W[i]
        cj += (r - W[i].dot(U[j]) - b[i] - mu)

      # set the updates
      U[j] = np.linalg.solve(matrix, vector)
      c[j] = cj / (len(movie2user[j]) + reg)

      if j % (M//10) == 0:
        print("j:", j, "M:", M)
    except KeyError:
      # possible not to have any ratings for a movie
      pass
  print("updated U and c:", datetime.now() - t0)
  print("epoch duration:", datetime.now() - epoch_start)


  # store train loss
  t0 = datetime.now()
  train_losses.append(loss(usermovie2rating))

  # store test loss
  test_losses.append(loss(usermovie2rating_test))
  print("calculate cost:", datetime.now() - t0)
  print("train loss:", train_losses[-1])
  print("test loss:", test_losses[-1])


print("train losses:", train_losses)
print("test losses:", test_losses)

# plot losses
plt.plot(train_losses, label="train loss")
plt.scatter(range(25),test_losses, label="test loss")
plt.legend()
plt.show()
t1 = datetime.now()