# Movie_Recommendation_System
  - A recommedation system is a simple algorithm whose aim is to provide the most relevent information to a user by dicovering patterns in a dataset. 
  - Basically this algorithm rates the items and it provides information such as 'What is the top rated movies' & provides recommendation based on some relavent     
    inforamation.
  - We have all used Netflix and Amazon and we are aware they provides recommendation based on the movies or web-series a user have already watched.
  - This engine makes suggestions by learning and understanding the patterns in your watch history and then applies those patterns and findings to make new 
    suggestions.
# Purpose of choosing this project
   - The main purpose of choosing this project was because of my own interest of exploring movie recommendation system.
# Libraries
  -requests, pandas, matplotlib, KNN (K-Nearest Neighbors), AST (Abstract Syntax Tree), cosine_similarity, JSON, KNN Imputer, CountVectorizer
# Dataset
  - Got the Dataset form TMDB website by creating developer account.
  - Made API request and got the API key. 
  - Created for loop to fatch movies from API key and converted into JSON format.
  - Converted JSON to csv.
# DataFrame
  - Created DataFrame of the csv file.
# Null Values/Datatypes
  - changed the data types of some features to the correct data types.
  - After finding null values from dataset replaced them by KNNImputer.
# Using AST
  - Created function and applied into dataframe and converted it into proper format which is easy to use.
# Top Trending Movies
  - Found the top 10 movies based on popularity
# Visualization
  - used powerbiclient, powerbiclient.authentication and Power BI Desktop, Power BI my workspace.
  - Made visualization in power bi desktop then published to myworkspace in power bi and called the report in jyupiter.
# Soup
  - Created soup by joining various features such as genres, language etc.
# Token
  - Created tokens using countvectorizer and cosine similarty.
# Recommendation Function
  - Created Recommendation Function using lambda function to get the movie recommendation.
