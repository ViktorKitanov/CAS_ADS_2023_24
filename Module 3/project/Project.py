import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import sklearn
#import tensorflow as tf
import pandas as pd
import umap
import plotly.express as px
#from IPython.display import display
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', True)

#Reading Publibike availability data
dfPubliBikeAvailability = pd.read_csv("data/bike-availability-All-Stations_hourly.csv", encoding='latin-1', sep=';')

#Prepareing Data
# TODO---> Vielleicht auch nur mal genau eine Woche nutzen zum trainieren der Daten, da eine Woche ein recht gutes Pattern ist.
dfPubliBikeAvailability = dfPubliBikeAvailability.rename(columns={"Abfragezeit": "timestamp"})
dfPubliBikeAvailability["timestamp"] = pd.to_datetime(dfPubliBikeAvailability["timestamp"])
dfPubliBikeAvailability["dayofweek"] = dfPubliBikeAvailability["timestamp"].dt.weekday
dfPubliBikeAvailability["hourofday"] = dfPubliBikeAvailability["timestamp"].dt.hour
# Create a continuous numerical representation for the x-axis
dfPubliBikeAvailability["continuous_time"] = dfPubliBikeAvailability['dayofweek'] * 24 + dfPubliBikeAvailability['hourofday']

# Filter the DataFrame based on the specified station ID
filtered_df = dfPubliBikeAvailability[dfPubliBikeAvailability['id'] == 230]

print(filtered_df)