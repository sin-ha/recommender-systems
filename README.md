# recommender-systems
<h3><u>building recommender systems that recommends movies</u></h3>

the dataset used in the project is the rating.csv file available in the kaggle competion linked below

https://www.kaggle.com/grouplens/movielens-20m-dataset
<u><h4>part 1</h4></u>
<h5>user-user collaborative filtering.py</h4>

in this we use the pearson corelation by calculating the measure of realation between any two person given their similarity in rating two movies thus improving our predictions from the normal average it is a non-optimised algorthm and takes <B>O(N*N +M) </B>time complexity to run hence a smaller subset of data  is considered in which the sparsity of the matrix can be reduced by only choosig the most frequent rating people and most frequently rated movies
