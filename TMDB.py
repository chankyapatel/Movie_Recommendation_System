#!/usr/bin/env python
# coding: utf-8

# # Recommendation System

# ### Libraries

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
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import json


# In[3]:


# To get the API
response = requests.get("https://api.themoviedb.org/3/movie/550?api_key=82860678603c30cb74ba5f32be02e2f7")


# # DataSet

# #### - This url contains only one movie data therefore, i will create a for loop to get more data from the url.

# In[4]:


# Checking the data
print(response.json())


# ### For Loop & Creating Json file

# In[5]:


# This loop will fatch more movies and also it will convert them into Json format
movie_id = list(range(11, 950)) # Here took 11 beacause 0-10 has not data of movie.

for i in range(len(movie_id)):
    url = 'https://api.themoviedb.org/3/movie/{}?api_key=82860678603c30cb74ba5f32be02e2f7'.format(movie_id[i])
    response = requests.get(url).json()
    text = json.dumps(response, sort_keys = True, indent = 4)
     #sort_keys = True tells the encoder to return the JSON object keys in a sorted order
    print(text)
    with open('TMDB.json', 'w') as json_file:  # This is where it is converting it into json file.
        json.dump(response, json_file)


# ## After getting all the data i have converted it into csv file.

# In[3]:


Movies_df = pd.read_csv("C:\\Users\\Patel Chankya\\Desktop\\Capstone Project\\TMDB.csv")


# In[4]:


Movies_df


# In[5]:


Movies_df.columns


# In[6]:


Movies_df.shape


# ## Dropping unnecessary features

# In[7]:


Movies_df.drop(labels = ['belongs_to_collection.name','adult','backdrop_path','belongs_to_collection.backdrop_path','belongs_to_collection.id','belongs_to_collection.poster_path','imdb_id','poster_path','status','tagline','video','status_code','status_message','success'], axis =1,inplace = True)


# In[8]:


Movies_df


# In[9]:


Movies_df = Movies_df.rename({'vote_average': 'Ratings'}, axis=1)


# ## Features

# In[10]:


Movies_df.columns


# ## Null Values

# In[11]:


Movies_df.isnull().sum()


# In[12]:


Nullvalues = Movies_df.isnull().sum()
Nullvalues = Nullvalues.to_csv("Nullvalues.csv",index = True)


# ## Dropping the blank rows from the dataset

# In[13]:


Movies_df = Movies_df.dropna(how = 'all')


# In[14]:


Movies_df.isnull().sum()


# In[18]:


Movies_df.shape


# In[15]:


Movies_df.describe()


# In[16]:


describe = Movies_df.describe()


# In[18]:


describe.to_csv("describe.csv",index = True)


# ## Rounding all the values

# In[17]:


Movies_df.describe().round(0)


# In[18]:


Movies_df


# # Data Types

# In[19]:


Movies_df.dtypes


# #### - Changing the release_date dtype to datetime

# In[20]:


Movies_df['release_date'] = pd.to_datetime(Movies_df['release_date'])


# In[21]:


Movies_df.dtypes


# ## Unique values

# In[22]:


import itertools


# In[23]:


import numpy as np


# In[24]:


np.unique([*itertools.chain.from_iterable(Movies_df.genres)])


# In[25]:


np.unique([*itertools.chain.from_iterable(Movies_df.spoken_languages)])


# In[26]:


np.unique([*itertools.chain.from_iterable(Movies_df.production_countries)])


# # Initial Dataset

# In[27]:


Movies_df1 = Movies_df[['genres','production_countries','spoken_languages']]
Movies_df1


# ### Using Abstract Syntax Grammer

# In[28]:


# AST it is a tree representation of the abstract syantatic structure of text written in a formal language.
import ast 


# #### - Defining get_dict_val containing a for loop which will return name.

# In[29]:


def get_dict_val(data):
    ls = []
    c =ast.literal_eval(data)
    for i in c:
        ls.append(i['name'])
    if len(ls)>0:
        return ls
    else: return ['Missing']


# #### - Applying get_dict_val in each column.

# In[30]:


Movies_df['genres']=Movies_df['genres'].apply(get_dict_val)


# In[31]:


# Here [6] is the 6th movie number
Movies_df['genres'][6]


# In[32]:


Movies_df['production_countries']=Movies_df['production_countries'].apply(get_dict_val)


# In[33]:


Movies_df['production_countries']


# In[34]:


Movies_df['spoken_languages']=Movies_df['spoken_languages'].apply(get_dict_val)


# In[35]:


Movies_df['spoken_languages']


# # Dataset after applying AST

# In[36]:


Movies_df


# ## Unique values after cleaning the data

# In[37]:


np.unique([*itertools.chain.from_iterable(Movies_df.genres)])


# In[38]:


np.unique([*itertools.chain.from_iterable(Movies_df.spoken_languages)])


# In[39]:


np.unique([*itertools.chain.from_iterable(Movies_df.production_countries)])


# # Top Trending Movies

# In[43]:


get_ipython().system(' pip install powerbiclient')


# In[40]:


from powerbiclient import Report, models


# In[41]:


from powerbiclient.authentication import DeviceCodeLoginAuthentication


# In[42]:


device_auth = DeviceCodeLoginAuthentication()


# In[43]:


group_id="50903f3f-667f-4268-abb7-82d4f439d40f"
report_id="5788edf0-013c-4df5-80b5-d613292a34f3"


# In[44]:


report = Report(group_id=group_id, report_id=report_id, auth=device_auth)

report


# # Ratings

# In[49]:


minvalue = Movies_df['Ratings'].min()


# In[50]:


minvalue


# In[51]:


maxvalue = Movies_df['Ratings'].max()


# In[52]:


maxvalue


# In[53]:


Movies_df.loc[Movies_df['title'] == "Scarface"]


# In[45]:


device_auth = DeviceCodeLoginAuthentication()


# In[46]:


group_id="390c6246-85e7-4101-8a19-b82b5b6ea0e9"
report_id="c21ed7bd-d825-4174-9b3d-d54465155931"


# #### - Top 10 Movies based on Highest ratings

# In[47]:


report1 = Report(group_id=group_id, report_id=report_id, auth=device_auth)

report1


# # Features

# In[48]:


features = ['spoken_languages','genres','overview','production_countries']


# #### - Now lets convert the above features into lowercase and remove all the spaces between them

# In[49]:


# Creating clean_data which contains if else loop to lower the case and remove the spaces between them.
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ""


# In[50]:


for feature in features:
    Movies_df[feature] = Movies_df[feature].apply(clean_data)


# # Soup

# In[51]:


# create_soup joins all the features.
def create_soup(x):
    return ' '.join(x['spoken_languages']) + ' ' + ' '.join(x['genres'])+ ' ' + ' '.join(x['overview'])+ ' ' + ' '.join(x['production_countries'])

# applying soup to Data Frame
Movies_df["soup"] = Movies_df.apply(create_soup, axis=1)
print(Movies_df["soup"].head())


# # indexing - it returns the index where the element appears in Data Frame

# In[52]:


indices = pd.Series(Movies_df.index, index=Movies_df["title"]).drop_duplicates()
print(indices.head())


# In[62]:


indices_2 = pd.Series(Movies_df.index, index=Movies_df["genres"]).drop_duplicates()
print(indices_2.head())


# # CountVectorizer

# #### - Creating tokens
# #### - Creating cosine_sim

# In[53]:


# stop words can be remove from data because they dont add any meaning to a sentence.
count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(Movies_df["soup"])

print(count_matrix.shape)

# Basically, cosine_sim contains similarities of all the features from soup.
cosine_sim = cosine_similarity(count_matrix, count_matrix)
print(cosine_sim.shape)

movies_df = Movies_df.reset_index() # it will reset the index
indices = pd.Series(movies_df.index, index=movies_df['title'])


# # Creating Recommendation Function

# In[54]:


# we will create a function named get_recommendations, which contains title and cosine similarity.
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    
# This sim_scores will make tuples where the first element is index and second element as cosine similarity score.
    sim_scores = list(enumerate(cosine_sim[idx]))
    
# This sim_score will sort the list of tuples in descending order based on the similarity score.
# lamda - lamda will create an inline function.
# x - the inline function will take x as an input and it willl return x[1].
# x[1] - it is the second element of x.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
# why 1? because indexing starts from 0 and it is the title of the movie itself.
    sim_scores = sim_scores[1:11] 
    
# Now we will map indices to the title.
    movies_indices = [ind[0] for ind in sim_scores]
    movies = Movies_df["title"].iloc[movies_indices]
    return movies # It will return the movie list


# # Recommendation

# In[57]:


name = str(input("Please enter the name of the movie: "))


# In[58]:


print(name)
print(get_recommendations(name))

