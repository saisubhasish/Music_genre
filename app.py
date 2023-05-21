import pandas as pd
from sklearn.cluster import KMeans
import folium
from flask import Flask, render_template
from music import utils
from data_dump import DATABASE_NAME, COLLECTION_NAME

import pandas as pd
from sklearn.cluster import KMeans
import folium
from flask import Flask, render_template
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score




df:pd.DataFrame  = utils.get_collection_as_dataframe(
                database_name=DATABASE_NAME, 
                collection_name=COLLECTION_NAME)

X = df.drop(['label', 'filename'], axis=1)

# Encode the labels for genre
le = LabelEncoder()
new_labels = pd.DataFrame(le.fit_transform(df['label']))
df['label'] = new_labels


# Standardizing data
scaler = StandardScaler()
features_normalized = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3)    # Building KMeans model with 3 clusters
kmeans.fit(features_normalized)

df['predicted_label'] = kmeans.labels_



accuracy = accuracy_score(le.inverse_transform(df['label']), le.inverse_transform(df['predicted_label']))

print('Accuracy of the model is: ',round(accuracy*100, 2))