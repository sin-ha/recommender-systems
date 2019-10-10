# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 19:29:29 2019

@author: HARSH
"""
import numpy as np
import pandas as pd 
dataset=pd.read_csv("rating.csv")
mod_data = dataset.drop(columns="timestamp")
"""user_data = mod_data["userId"].value_counts().iloc[:2500]
movie_data = mod_data["movieId"].value_counts().iloc[:2500]
t =[]
i = 0
l = []
usertomovie = {}
movietouser = {}
movieusertorating = {}
for elem1,elem2 in zip(user_data,movie_data):
    
for elem ,elem1 in zip(user_data.index,user_data):
    case_str = "userId==" + str(elem)
    t.append(mod_data.query(case_str)["rating"].sum())
    l.append(t[i]/elem1)
    i+=1
    if(i%100==0):
        print(i)"""
mod_data.userId -= 1
unique_movies = set(mod_data.movieId)
coded_movie = {}
count=0
for movie in unique_movies:
    coded_movie[movie] = count
    count+=1
mod_data["movieId"] = mod_data.apply(lambda row: coded_movie[row.movieId], axis=1)
mod_data.to_csv("cleaned1",index =False)

user_data =mod_data["userId"].value_counts().iloc[:2500].index.values
movie_data = mod_data["movieId"].value_counts().iloc[:2500].index.values
#<----------------------------shrinking the dataset------------------------>

df  = pd.read_csv("editing1.csv")

M = df.userId.max()+1

N = df.movie_idx.max()+1

from collections import Counter
n = 10000
m = 2500
user_id_max = Counter(df.userId)
movie_id_max = Counter(df.movie_idx)

user_list  = [u for u,c in user_id_max.most_common(n)]
movie_list = [u for u,c in movie_id_max.most_common(m)]

df_small =df[df.movie_idx.isin(movie_list) & df.userId.isin(user_list)].copy()
new_user_id_map = {}
i = 0
for old in user_list:
  new_user_id_map[old] = i
  i += 1
print("i:", i)

new_movie_id_map = {}
j = 0
for old in movie_list:
  new_movie_id_map[old] = j
  j += 1
print("j:", j)

#<-----------------------------creatinng dictionaries----------------------->
from random import shuffle
df = df_small.copy()
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

user2movie = {}
movie2user  = {}
usermovie2rating = {}

count = 0
def update(row):
  
  global count 
  count += 1
  if count % 100000 == 0:
    print("processed: %.3f" % (float(count)/cutoff))

  i = int(row.userId)
  j = int(row.movie_idx)
  if i not in user2movie:
    user2movie[i] = [j]
  else:
    user2movie[i].append(j)

  if j not in movie2user:
    movie2user[j] = [i]
  else:
    movie2user[j].append(i)

  usermovie2rating[(i,j)] = row.rating

df_train.apply(update,axis =1)    
    
usermovie2rating_test = {}
print("Calling: update_usermovie2rating_test")
count = 0
def update_usermovie2rating_test(row):
  global count
  count += 1
  if count % 100000 == 0:
    print("processed: %.3f" % (float(count)/len(df_test)))

  i = int(row.userId)
  j = int(row.movie_idx)
  usermovie2rating_test[(i,j)] = row.rating
df_test.apply(update_usermovie2rating_test, axis=1)
#<---------------------------------------creating the model------------------------------>
from sortedcontainers import SortedList
K = 25 # number of neighbors we'd like to consider
limit = 5 # number of common movies users must have in common in order to consider
N = max(list(user2movie.keys()))+1
m1 =  max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1,m2)+1
neighbors = [] # store neighbors in this list
averages = [] # each user's average rating for later use
deviations = [] # each user's deviation for later use

for i in range(N):
    movies_i = user2movie[i]
    movies_i_set  = set(movies_i)
    
    ratings_i = {movie : usermovie2rating[(i, movie)] for movie in movies_i}
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = {movie : (usermovie2rating[(i,movie)]-avg_i) for movie in movies_i}
    dev_i_values = np.array(list(dev_i.values()))
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))
    
    averages.append(avg_i)
    deviations.append(dev_i)
    sl = SortedList()
    for j in range(N):
        if(i!=j):
            movies_j = user2movie[j]
            movies_j_set = set(movies_j)
            common_movies = (movies_i_set & movies_j_set) # intersection
            if len(common_movies) > limit:
                ratings_j = {movie : usermovie2rating[(j, movie)] for movie in movies_j}
                avg_j = np.mean(list(ratings_j.values()))
                dev_j = {movie : (usermovie2rating[(j,movie)]-avg_j) for movie in movies_j}
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))
                
                numerator = sum(dev_i[m]*dev_j[m] for m in common_movies)
                w_ij = numerator / (sigma_i * sigma_j)
                sl.add((-w_ij, j))
                if len(sl) > K:
                  del sl[-1]
    neighbors.append(sl)
    if i % 1 == 0:
        print(i)
""" ts       
424 22 24
564 22 44
649 22 55
685 23 00
931 23 35
960 23 37



"""
        
    
    
    