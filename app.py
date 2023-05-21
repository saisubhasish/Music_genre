import pandas as pd
from sklearn.cluster import KMeans
import folium
from flask import Flask, render_template
from music import utils
from data_dump import DATABASE_NAME, COLLECTION_NAME




df:pd.DataFrame  = utils.get_collection_as_dataframe(
                database_name=DATABASE_NAME, 
                collection_name=COLLECTION_NAME)

print(df)