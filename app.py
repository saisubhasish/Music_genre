from flask import Flask, render_template
import folium
from music import utils
from music.config import database_name, collection_name



app = Flask(__name__)


df = utils.get_collection_as_dataframe(
                database_name = database_name, 
                collection_name = collection_name)

def create_map(df):
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
    
    # Add markers for each data point
    for index, row in df.iterrows():
        folium.Marker([row['latitude'], row['longitude']], popup=row['predicted_label']).add_to(m)
    
    return m

@app.route('/')
def show_map():
    m = create_map(df)
    return render_template('map.html', map=m._repr_html_())



if __name__ == '__main__':
    app.run()