'''
A recommendation engine, also known as a recommender system, 
is software that analyzes available data to make suggestions for something that a user might be interested in.

There are basically four or three types of recommendation engines depending on who you ask:
1.Content based recommendation engine
2.Collaborative filtering based recommendation engine
3.Popularity based recommendation engine
4.Hybrid recommendation engine

The engine we will use is called a content based recommendation engine and it is 
a recommendation system that takes in a movie that a user likes 
and then analyzes it to get the movies content (e.g. genre, cast, director, keywords, etc.), 
it then ranks the recommended movies based on how similar the recommended movies 
are to the liked movie using something called similarity scores
'''

#Description: Build a movie recommendation engine (more specifically a content based recommendation engine)

#Resources: https://medium.com/code-heroku/building-a-movie-recommendation-engine-in-python-using-scikit-learn-c7489d7cb145

#Import Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Load the data
df = pd.read_csv("movie_dataset.csv")


#Create a list of important columns to keep a.k.a. the main content of the movie
features = ['keywords','cast','genres','director']


#Clean and preprocess the data
for feature in features:
    df[feature] = df[feature].fillna('') #Fill any missing values with the empty string
   # print(df[feature])


#A function to combine the values of the important columns into a single string
def combine_features(row):
    return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]


#Apply the function to each row in the dataset to store the combined strings into a new column called combined_features 
df["combined_features"] = df.apply(combine_features,axis=1)
#df["combined_features"]


#Convert a collection of text to a matrix/vector of token counts
count_matrix = CountVectorizer().fit_transform(df["combined_features"])


#Get the cosine similarity matrix from the count matrix (cos(theta))
cosine_sim = cosine_similarity(count_matrix)


#Helper function to get the title from the index
def get_title_from_index(index):
  return df[df.index == index]["title"].values[0]
#Helper function to get the index from the title
def get_index_from_title(title):
  return df[df.title == title]["index"].values[0]


#Get the title of the movie that the user likes
print("Enter the name of the movie")
movie_user_likes = input()

#Find that movies index
movie_index = get_index_from_title(movie_user_likes)

#Access the row, through the movies index, corresponding to this movie (the liked movie) in the similarity matrix, 
# by doing this we will get the similarity scores of all other movies from the current movie

#Enumerate through all the similarity scores of that movie to make a tuple of movie index and similarity scores.
#  This will convert a row of similarity scores like this- [5 0.6 0.3 0.9] to this- [(0, 5) (1, 0.6) (2, 0.3) (3, 0.9)] . 
#  Note this puts each item in the list in this form (movie index, similarity score)
similar_movies =  list(enumerate(cosine_sim[movie_index]))



#Sort the list of similar movies according to the similarity scores in descending order
#Since the most similar movie is itself, we will discard the first element after sorting.
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]


#Create a loop to print the first 5 entries from the sorted similar movies list

i=0
print("Top 5 similar movies to "+movie_user_likes+" are:")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]) )
    i=i+1
    if i>=5:
        break

        