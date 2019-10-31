# recommender-systems
<h3><u>building recommender systems that recommends movies</u></h3>

the dataset used in the project is the rating.csv file available in the kaggle competion linked below

https://www.kaggle.com/grouplens/movielens-20m-dataset
<u><h4>part 1</h4></u>
<h5>user-user collaborative filtering.py</h4>

in this we use the pearson corelation by calculating the measure of realation between any two person given their similarity in rating two movies thus improving our predictions from the normal average it is a non-optimised algorthm and takes <B>O(N*N +M) </B>time complexity to run hence a smaller subset of data  is considered in which the sparsity of the matrix can be reduced by only choosig the most frequent rating people and most frequently rated movies
this algorithm is highly compute intensive and hence a only 10k most frequent users and 2.5k most rated movies are accounted for 
In each iteration the similarity is calculated between two users by the pearson corelation and the top 5 neighbours are stored only

## improvement 
since this is a mathematical model which is deterministic hence we improve it by considering more neighbours which give us a better estimation

<u><h4>part 2</h4></u>
<h5>matrix factorization.py</h4>
Based on SVD we try to produce our rating matrix as a product of two latent space matrices U and W 
the predicted value is given as r<sup>^</sup>  = W<sup>T</sup>.U + b +c +mu
and hence the gradients and cost function
