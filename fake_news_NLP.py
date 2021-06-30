# Natural Language Processing: Identifying Fake News
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import pickle

df_news = pd.read_csv("news.csv")
# df_news.head()

# df_news.info()

# df_news.isnull().sum()

features = ["text"]
label = ["label"]
X, y = df_news[features].values, df_news[label].values
# print(y.shape)

# train_test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# X_test[6]

# preprocess natural language data using TfidfVectorizer function
""" Initialize TDFIDF with stop words from the English language 
and filter out terms with maximum document frequency greater than 0.7"""

TDIDF_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit train set, and transform test set
X_train_transform = TDIDF_vectorizer.fit_transform(X_train.ravel())
X_test_transform  = TDIDF_vectorizer.transform(X_test.ravel())

# Fit model using the PassiveAggresiveClassifier
model = PassiveAggressiveClassifier().fit(X_train_transform, y_train.ravel())

# Model evaluation
y_predict = model.predict(X_test_transform)
score = accuracy_score(y_test, y_predict)
print(f"Accuracy : {round(score*100,2)} %")

# Confusion Matrix
confusion_df = pd.DataFrame(data = confusion_matrix(y_test, y_predict), index = ['positive_actual', 'negative_actual'], columns = ['positive_predicted', 'negative_predicted'])
# confusion_df

"""# save models"""

import pickle
# save vectorizer
with open("TDIDF_vectorizer.pkl", "wb") as outfile:
    TDIDF_vectorizer = pickle.dump(TDIDF_vectorizer, outfile)

# save model
with open('model.pkl', 'wb') as outfile:
    pickle.dump(model, outfile)

