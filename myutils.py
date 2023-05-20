# Import all the required libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import panel as pn
pn.extension('plotly')
import hvplot.pandas
import holoviews as hv

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
# Library that has a lot of operating system functions
from os import getenv
import requests
import time

import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

import string
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import pymongo
import json
import param
# pd.options.plotting.backend = 'holoviews'

# function to perform regression using specified model
def perform_regression(model, X, y):
    # Divide the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit a multiple linear regression model using Scikit-learn
    regression_model = model.fit(X_train, y_train)
    
    # Evaluate the performance of the model using the testing set
    train_score = regression_model.score(X_train, y_train)
    test_score = regression_model.score(X_test, y_test)
    return [model, train_score, test_score]
#     print("Train R-squared score:", train_score)
#     print("Test R-squared score:", test_score)
    
#     y_pred = regression_model.predict(X_test)
#     residuals = y_test - y_pred

    # Display plots
#     fig, axes = plt.subplots(1,2, figsize=(30, 6), sharex=True)
#     axes[0].scatter(y_pred, residuals, label=str(model))
#     axes[0].axhline(y=0, color='r', linestyle='-')
#     axes[0].set_xlabel('Predicted Values')
#     axes[0].set_ylabel('Residuals')
#     axes[1].scatter(y_test, y_pred, label=str(model))
#     axes[1].plot(y_test, y_test, color='black', linestyle='--', label='Perfect prediction')
#     axes[1].set_xlabel('Actual Values')
#     axes[1].set_ylabel('Predicted Values')
#     axes[0].legend()
#     axes[1].legend()
#     plt.show()
    

# Function to identify the most important features
# Returns the list of features ordered by their magnitudes
def get_important_features_with_lasso(X, y):
    # Scaling the data
    X_scaled = StandardScaler().fit_transform(X)
    lasso = Lasso(max_iter=100000).fit(X_scaled, y)

    # Get the absolute values of coefficients from the model
    coefficients = pd.Series(np.abs(lasso.coef_), index=X.columns)
    # Print the top 15 coefficients with highest magnitude value
    ordered_features = coefficients.sort_values(ascending=False)
    
    return ordered_features

# Function to define hyperparameter grid
def get_param_grid(model):
    # Define hyperparameter grid
    if model.__class__.__name__ in ['Ridge', 'Lasso']:
        param_grid = {'alpha': np.logspace(-3, 3, 7)}
        if model.__class__.__name__ == 'Lasso':
            param_grid['max_iter'] = [10000]
    elif model.__class__.__name__ == 'LogisticRegression':
        param_grid = {
            'penalty': ['l1', 'l2'],        # L1->Lasso, L2->Ridge
            'C': [0.1, 1.0],                # inverse of regularization strength
            'solver': ['liblinear', 'saga'], # Optimzation Algorithm
            'max_iter': [10000]
        }
    elif model.__class__.__name__ == 'DecisionTreeClassifier':
        param_grid = {
            'criterion': ['gini', 'entropy'],   # Quailty of split: gini or entropy
            'max_depth': [None, 5, 10, 15],     # Depth of the tree
#             'min_samples_split': [2, 5, 10],    # 
#             'min_samples_leaf': [1, 2, 4],
#             'max_features': [None, 'sqrt', 'log2']
        }
    elif model.__class__.__name__ == 'RandomForestClassifier':
        param_grid = {
            'n_estimators': [100, 200, 300],    # Number of Decision Trees in the random forest
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10],
#             'min_samples_split': [2, 5, 10],
#             'min_samples_leaf': [1, 2, 4],
#             'max_features': ['sqrt', 'log2']
        }
    elif model.__class__.__name__ == 'SVC':
        param_grid = {
            'C': [0.1, 1, 10],    # Penalty for misclassifying data points
            'kernel': ['linear', 'rbf', 'sigmoid'], # Kernel function to map input space into higher-dimensional feature space
            'gamma': ['scale', 'auto']   # Kernel coefficient for above rbf and sigmoid kernels
        }
    else:
        param_grid = {
            'fit_intercept': [True, False]
        }
    return param_grid

# Function to filter the dependent features based on user input
def get_selected_features(X, ordered_features, feature):
    if feature == 'Top 5':
        X = X[ordered_features[:6].keys()]
    elif feature == 'Top 10':
        X = X[ordered_features[:11].keys()]
    elif feature == 'Top 15':
        X = X[ordered_features[:16].keys()]
    elif feature == 'Top 20':
        X = X[ordered_features[:21].keys()]
    elif feature == 'Top 25':
        X = X[ordered_features[:26].keys()]
    return X

# Function to run defined model and return results
def get_model_visualization_results(joined_df, model, scaler, gridsearchcv, feature):
    # Define Classification models
    clf_models = ["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "SVC"]
    
    # Define indepedent features
    X = joined_df.iloc[:, :-6]  # All Construction materials
    
    # Initiate lists to store all train and test scores for visualization
    train_scores =[]
    test_scores = []
    
    # Make 2 subplots
#     fig = make_subplots(rows=1, cols=2, vertical_spacing=0.25)
    scatter_scores = []
    results = []
    scatter_plots = []
    heatmap_plots = []
    # Loop through each building/construction type
    building_types = joined_df.columns[-5:]
    for dep_col in building_types:              
        # Check if Classification model is chosen.
        if model.__class__.__name__ in clf_models:
            # Compare the value with previous quarter, and set it to 1 if true else to 0
            y = (joined_df[dep_col] > joined_df[dep_col].shift()).astype('int').fillna(0).values
        else:
            # Set the target type to building type
            y = joined_df[dep_col]
            
        # Feature Selection
        if feature != 'All':
            ordered_features = get_important_features_with_lasso(X, y)
            X = get_selected_features(X, ordered_features, feature)
                
        # Divide the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale data if selected
        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            
        # GridSearchCV hyper-parameter tuning and cross-validation
        if gridsearchcv:
            # Define hyperparameter grid
            param_grid = get_param_grid(model)
                
            # Define GridSearchCV object
            # Setting n_jobs=-1 to utilize all available cores, else it can take long for LogisticRegression
            grid_search = GridSearchCV(model, param_grid, cv=5, return_train_score=True, scoring='r2', n_jobs=-1) 

            # Fit GridSearchCV object to data
            grid_search.fit(X_train, y_train)
    
            # Use the best estimator Model
            model = grid_search.best_estimator_

        # else:
        # Perform normal regression with specified model
        regression_model = model.fit(X_train, y_train)
        
        # Predict the target values using the trained model
        y_pred = regression_model.predict(X_test)

        # Evaluate the performance of the model using the testing set
        train_score = regression_model.score(X_train, y_train)
        test_score = regression_model.score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        results.extend([
            {'Building Type': dep_col, 'Score Type': 'Training', 'Score': train_score},
            {'Building Type': dep_col, 'Score Type': 'Test', 'Score': test_score}
        ])
        
        # Check if Regression or Classification model is being used
        if model.__class__.__name__ in clf_models:
            # Calculate confusion matrix for the classifier
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
            heatmap_plots.append(cm_df.hvplot.heatmap(label=dep_col, width=500))
        else:
            # Scatter plot for regression models to show actual and predicted values
            scatter_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
            scatter_plots.append(scatter_df.hvplot.scatter(x='y_test', y='y_pred', label=dep_col, title='Actual vs Predicted Values', width=900))
            perfect_line = pd.DataFrame({'y_test': y_test, 'y_pred': y_test})
            scatter_plots.append(perfect_line.hvplot.line(x='y_test', y='y_pred', width=900))

    result_df = pd.DataFrame(results)
    fig1 = result_df.hvplot.bar(x='Building Type', y='Score', by='Score Type', title=f"Model:{model}, Scaler:{scaler}, GridSearchCV:{gridsearchcv}, Features:{feature}", width=1000)
#     fig1.opts(xrotation=45)  # Rotate x-axis labels by 45 degrees

    # Check whether regression or classification model
    if model.__class__.__name__ in clf_models:
        # Plot heatmaps for confusion matrix
        fig2 = heatmap_plots[0]
        for plot in heatmap_plots[1:]:
            fig2 += plot
    else:
        # Plot scatter the actual and predicted values
        fig2 = scatter_plots[0]
        for plot in scatter_plots[1:]:
            fig2 *= plot
    return hv.Layout(fig1 + fig2).cols(1)

# Function to search Reddit posts
def search_reddit(headers, subreddit, keyword):
    # Searching for a keyword within sub-reddit for last 1 year and sort by comments
    # https://www.reddit.com/dev/api#GET_search
    params = {'limit': 100, 'restrict_sr': 1, 'sr_nsfw': 1, 't': 'all', 'sort': 'comments'}
    search_results = []

    for i in range(100):
        res = requests.get(
            f"https://oauth.reddit.com/r/{subreddit}/search?q={keyword}",
            headers=headers,
            params=params
        )

        results = res.json()['data']['children']
        search_results.extend(results)

        # Get the ID of last post and start next search after it
        try:
            fullname = f"{results[-1]['kind']}_{results[-1]['data']['id']}"
            # This tells the API that the post we want is the one after the current one in the queue (ie is the next oldest one).
            params['after'] = fullname
            # Sleeping for 1 second to not exceed Reddit's API limits
            time.sleep(1)
        except:
            print(f"No search results after last result ID: {params['after']}")
            break

    # Total search results for the keyword
    print(f"Search results: {len(search_results)}")
    return get_comments_for_posts(headers, search_results)


# Function to get comments/replies for each post
def get_comments_for_posts(headers, search_results):
    # Define a list to store all the post and comment responses
    comments = []
    for post in search_results:
        # Get comments for each post
        res = requests.get(
            f"https://oauth.reddit.com{post['data']['permalink']}",
            headers=headers
        )
        # Add the retreived post body
        post = res.json()[0]['data']['children'][0]['data']
        replies = res.json()[1]['data']['children']
        comments.append({'created_utc': post['created_utc'],
                      'author': post['author'],
                      'post_id': post['id'],
                      'subreddit': post['subreddit'],
                     'text': post['selftext']})
        # Add the retreived comments to "comments" list
        # Loop through the comments and extract the reply bodies
        for comment in replies:
            reply_bodies = extract_reply_body(comment)
            if reply_bodies:
                comments.append(reply_bodies)
        
        # Sleeping for 0.1 second to not exceed Reddit's API limits
        time.sleep(0.1)

    print(f"Total comments retreived: {len(comments)}")
    return comments
    
# Define a function to extract the reply body from a comment or a reply
# It takes reply as input and returns its body or list of bodies
def extract_reply_body(reply):
    if reply['kind'] == 't1':
        # If the reply is a comment, extract the comment body
        return {
            'created_utc': reply['data']['created'],
            'author': reply['data']['author'],
            'post_id': reply['data']['link_id'].split('_')[1],
            'subreddit': reply['data']['subreddit'],
            'text': reply['data']['body']
        }
    elif reply['kind'] == 'Listing':
        # If the reply is a listing of replies, traverse the listing and extract the reply bodies
        reply_bodies = []
        for child in reply['data']['children']:
            reply_body = extract_reply_body(child)
            if reply_body:
                reply_bodies.append(reply_body)
        return reply_bodies

# Function to clean data before Sentiment Analysis
def get_cleaned_data(X):
    # Store the stopwords into the object named as "stop_words"
    stop_words = stopwords.words('english')

    # Store the string.punctuation into an object punct
    punct = string.punctuation

    # Initialise an object using a method PorterStemmer
    stemmer = PorterStemmer()

    cleaned_data=[]

    # For loop from first value to length(X), ^a-zA-Z means include small and capital case letters
    for i in range(len(X)):
        post = re.sub('[^a-zA-Z]', ' ', X.iloc[i])
        post = post.lower().split()
        post = [stemmer.stem(word) for word in post if (word not in stop_words) and (word not in punct)]
        post = ' '.join(post)
        cleaned_data.append(post)

    return pd.Series(cleaned_data)


# Function to classify compound result 
def classify_sia_polarity(output_dict):  
    polarity = "neutral"

    if(output_dict['compound'] >= 0.05):
        polarity = "positive"

    elif(output_dict['compound'] <= -0.05):
        polarity = "negative"

    return polarity

# Function to classify textblob's sentiment polarity
def classify_textblob_polarity(polarity):
    sentiment = "neutral"
    if(polarity > 0):
        sentiment = "positive"
    elif(polarity < 0):
        sentiment = "negative"
    return sentiment

# Function to predict sentiment
def predict_sentiment(text): 
    output_dict =  SentimentIntensityAnalyzer().polarity_scores(text)
    return classify_sia_polarity(output_dict)

# Function to calculate sentiment polarity using TextBlob
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = classify_textblob_polarity(blob.sentiment.polarity)
    return sentiment


