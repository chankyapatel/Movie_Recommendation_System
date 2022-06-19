#!/usr/bin/env python
# coding: utf-8

# # Libraries

# #### - request allows us to send HTTP requests using python. The HTTPS requests returns a response object like data.
# #### - The cosine similarity is used for matching similar documents based on maximum number of common words in documents.
# #### - CountVectorizer is used to convert a collectoin of text into tokens.
# #### - linear_kernel can be use to saperate data using a single line.
# #### - TfidfVectorizer is used to find the most frequent words. For this it uses in-memory vocabulary.

# In[1]:


import requests 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


import json


# In[3]:


response = requests.get("https://api.themoviedb.org/3/movie/550?api_key=82860678603c30cb74ba5f32be02e2f7")


# # DataSet

# #### - This url contains only one movie data therefore, i will create a for loop to get more data from the url.

# In[4]:


print(response.json())


# ### For Loop

# In[5]:


movie_id = list(range(11, 950))

for i in range(len(movie_id)):
    url = 'https://api.themoviedb.org/3/movie/{}?api_key=82860678603c30cb74ba5f32be02e2f7'.format(movie_id[i])
    response = requests.get(url).json()
    text = json.dumps(response, sort_keys = True, indent = 4)
    print(text)
    with open('TMDB.json', 'w') as json_file:
        json.dump(response, json_file)


# ## After getting all the data i have converted it into csv file.

# In[6]:


Movies_df = pd.read_csv("C:\\Users\\Patel Chankya\\Desktop\\Capstone Project\\TMDB.csv")


# In[7]:


Movies_df


# ## Dropping unnecessary features

# In[8]:


Movies_df.drop(labels = ['belongs_to_collection.name','adult','backdrop_path','belongs_to_collection.backdrop_path','belongs_to_collection.id','belongs_to_collection.poster_path','imdb_id','poster_path','status','tagline','video','status_code','status_message','success'], axis =1,inplace = True)


# In[9]:


Movies_df


# ## Features

# In[10]:


Movies_df.columns


# ## Null Values

# In[11]:


Movies_df.isnull().sum()


# In[12]:


Movies_df.describe()


# ## Rounding all the values

# In[13]:


Movies_df.describe().round(0)


# # Replacing Null Values

# #### - Replacing all the null values with KNN(K-Nearest Neighbors)

# In[14]:


from sklearn.impute import KNNImputer


# In[15]:


import numpy as np


# In[16]:


imputer = KNNImputer(n_neighbors=5)


# In[17]:


movies = ["vote_average", "vote_count","popularity","budget","revenue"]


# In[18]:


Movies_df[movies].isnull().sum()


# In[19]:


A = Movies_df[movies]


# In[20]:


Movies = pd.DataFrame(imputer.fit_transform(A),columns = A.columns)


# In[21]:


Movies.isnull().sum()


# #### - Droppping all the unnecessary null values from DataSet

# In[22]:


Movies_df = Movies_df.dropna(subset=['genres','id','original_language','original_title','overview','production_companies','production_countries','release_date','runtime','spoken_languages'])


# # Data Types

# In[23]:


Movies_df.dtypes


# #### - Changing the release_date dtype to datetime

# In[24]:


Movies_df['release_date'] = pd.to_datetime(Movies_df['release_date'])


# In[25]:


Movies_df.dtypes


# # Initial Dataset

# In[26]:


Movies_df1 = Movies_df[['genres','production_countries','spoken_languages']]
Movies_df1


# ### Using Abstract Syntax Grammer

# In[27]:


import ast 


# #### - Defining get_dict_val containing a for loop which will return name.

# In[28]:


def get_dict_val(data):
    ls = []
    c =ast.literal_eval(data)
    for i in c:
        ls.append(i['name'])
    if len(ls)>0:
        return ls
    else: return ['Missing']


# #### - Applying get_dict_val in each column.

# In[29]:


Movies_df['genres']=Movies_df['genres'].apply(get_dict_val)


# In[30]:


Movies_df['genres'][6]


# In[31]:


Movies_df['production_countries']=Movies_df['production_countries'].apply(get_dict_val)


# In[32]:


Movies_df['production_countries']


# In[33]:


Movies_df['spoken_languages']=Movies_df['spoken_languages'].apply(get_dict_val)


# In[34]:


Movies_df['spoken_languages']


# # Dataset after applying AST

# In[35]:


Movies_df


# # Top Trending Movies

# In[36]:


import matplotlib.pyplot as plt


# In[37]:


# Plot top 10 movies
# for every user
def plot():
    popularity = Movies_df.sort_values("popularity", ascending=False)
    plt.figure(figsize=(12, 6))
    plt.barh(popularity["title"].head(10), popularity["popularity"].head(10), align="center", color="green")
    plt.gca().invert_yaxis()
    plt.title("Top Trending movies")
    plt.xlabel("Popularity")
    plt.show()
    

plot()


# # Weighted Ratings

# #### - Creating 2 vectors one contains mean of "Vote_average" and second contain at 0.9 quantile of "Vote_count"

# In[38]:


C = Movies_df["vote_average"].mean()
m = Movies_df["vote_count"].quantile(0.9)

print("C: ", C)
print("m: ", m)

new_Movies_df = Movies_df.copy().loc[Movies_df["vote_count"] >= m]
print(new_Movies_df.shape)


# #### - Weighted ratings works on a mathematical formula

# In[39]:


def weighted_rating(x, C=C, m=m):
    v = x["vote_count"]
    R = x["vote_average"]

    return (v/(v + m) * R) + (m/(v + m) * C)


# # Ratings

# #### - Top 10 Movies based on Highest ratings

# In[40]:


new_Movies_df["Ratings"] = new_Movies_df.apply(weighted_rating, axis=1)
new_Movies_df = new_Movies_df.sort_values('Ratings', ascending=False)

new_Movies_df[["original_title", "vote_count", "vote_average", "Ratings"]].head(10)


# # Features

# In[41]:


features = ['spoken_languages','genres','overview','production_countries']


# #### - Now lets convert the above features into lowercase and remove all the spaces between them

# In[42]:


# Creating clean_data which contains if else loop to lower the case and remove the spaces between them.
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ""


# In[43]:


for feature in features:
    Movies_df[feature] = Movies_df[feature].apply(clean_data)


# # Soup

# In[44]:


# create_soup joins all the features.
def create_soup(x):
    return ' '.join(x['spoken_languages']) + ' ' + ' '.join(x['genres'])+ ' ' + ' '.join(x['overview'])+ ' ' + ' '.join(x['production_countries'])

# applying soup to Data Frame
Movies_df["soup"] = Movies_df.apply(create_soup, axis=1)
print(Movies_df["soup"].head())


# ### - Now we will create a reverse mapping of title to indices. So that we can easily find the title based on the index

# In[49]:


# index - it returns the index where the element appears in Data Frame
indices = pd.Series(Movies_df.index, index=Movies_df["title"]).drop_duplicates()
print(indices.head())


# # CountVectorizer

# #### - Creating tokens
# #### - Creating cosine_sin

# In[50]:


# stop words can be remove from data because they dont add meaning to a sentence.
count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(Movies_df["soup"])

print(count_matrix.shape)

# Basically, cosine_sim contains similarities of all the features from soup.
cosine_sim = cosine_similarity(count_matrix, count_matrix)
print(cosine_sim.shape)

movies_df = Movies_df.reset_index()
indices = pd.Series(movies_df.index, index=movies_df['title'])


# # Creating Recommendation

# In[57]:


# we will create a function named get_recommendations, which will take title and cosine similarity score as an input.
def get_recommendations(title, cosine_sim=cosine_sim):
    """
    in this function,
        we take the cosine score of given movie
        sort them based on cosine score (movie_id, cosine_score)
        take the next 10 values because the first entry is itself
        get those movie indices
        map those indices to titles
        return title list
    """
    idx = indices[title]
# This sim_scores will make tuples where the first element is index and second element as cosine similarity score.
    sim_scores = list(enumerate(cosine_sim[idx]))
# This sim_score will sort the list of tuples in descending order based on the similarity score.
# lamda - lamda will create an inline function.
# x - the inline function will take x as an input and it willl return x[1].
# x[1] - it is the second elemnt of x.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
# why 1? because indexing starts from 0 and it is the title of the movie itself.
    sim_scores = sim_scores[1:11] 
# Now we will map indices to the title.
    movies_indices = [ind[0] for ind in sim_scores]
    movies = Movies_df["title"].iloc[movies_indices]
    return movies # It will return the movie list


# In[91]:


print("Finding Nemo")
print(get_recommendations("Finding Nemo"))
print()

