import numpy as np
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import sklearn
#import tensorflow as tf
import pandas as pd
import umap
import plotly.express as px
#from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier, plot_tree

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', True)

#Reading Publibike availability data
dfPubliBikeAvailability = pd.read_csv("data/bike-availability-All-Stations_hourly.csv", encoding='latin-1', sep=';')

#Prepareing Data
dfPubliBikeAvailability = dfPubliBikeAvailability.rename(columns={"Abfragezeit": "timestamp"})
dfPubliBikeAvailability["timestamp"] = pd.to_datetime(dfPubliBikeAvailability["timestamp"])
dfPubliBikeAvailability["dayofweek"] = dfPubliBikeAvailability["timestamp"].dt.weekday
dfPubliBikeAvailability["hourofday"] = dfPubliBikeAvailability["timestamp"].dt.hour
# Create a continuous numerical representation for the x-axis
dfPubliBikeAvailability["continuous_time"] = dfPubliBikeAvailability['dayofweek'] * 24 + dfPubliBikeAvailability['hourofday']

#Assigning the availability into 3 Groups;  Group "0" --> Available bikes = 0-1; Group "1" --> Available bikes = 2-4; Group "2" --> 5 or more
dfPubliBikeAvailability['bike_availability'] = dfPubliBikeAvailability['Bike']
dfPubliBikeAvailability['bike_availability'] = [0 if (i<2) else i for i in dfPubliBikeAvailability['bike_availability']]
dfPubliBikeAvailability['bike_availability'] = [1 if (1<i<5) else i for i in dfPubliBikeAvailability['bike_availability']]
dfPubliBikeAvailability['bike_availability'] = [2 if (i>4) else i for i in dfPubliBikeAvailability['bike_availability']]

dfPubliBikeAvailability['e-bike_availability'] = dfPubliBikeAvailability['EBike']
dfPubliBikeAvailability['e-bike_availability'] = [0 if (i<2) else i for i in dfPubliBikeAvailability['e-bike_availability']]
dfPubliBikeAvailability['e-bike_availability'] = [1 if (1<i<5) else i for i in dfPubliBikeAvailability['e-bike_availability']]
dfPubliBikeAvailability['e-bike_availability'] = [2 if (i>4) else i for i in dfPubliBikeAvailability['e-bike_availability']]

# Filter the DataFrame based on the specified station ID and desired date range
start_date = '2023-05-15'
end_date = '2023-09-15'
filtered_df = dfPubliBikeAvailability[(dfPubliBikeAvailability['timestamp'] >= start_date) &
                                      (dfPubliBikeAvailability['timestamp'] <= end_date) &
                                      (dfPubliBikeAvailability['id'] == 230)]

# With a dataframe with columns 'x', and 'y'
#x = filtered_df[["continuous_time"]]
x = filtered_df[["dayofweek", "hourofday"]]
y = filtered_df['bike_availability']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# make 3-class dataset for classification

transformation = [[0.4, 0.2], [-0.4, 1.2]]
x_train_transformed = np.dot(x_train, transformation)

dtcs = []
for depth in (1, 2, 3, 4, 5):
    # do fit
    dtc = DecisionTreeClassifier(max_depth=depth, criterion='entropy')  # 'entropy'
    dtcs.append(dtc)
    dtc.fit(x_train_transformed, y_train)

    # print the training scores
    print("Training score : %.3f (depth=%d)" % (dtc.score(x_train, y_train), depth))

    fig, ax = plt.subplots(1, 2,  figsize=(8,4), dpi=150)

    # Plot decision boundaries
    h = 0.02
    x_min, x_max = x_train_transformed[:, 0].min() - 1, x_train_transformed[:, 0].max() + 1
    y_min, y_max = x_train_transformed[:, 1].min() - 1, x_train_transformed[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = dtc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax[0].contourf(xx, yy, Z, alpha=0.5)

    # Plot training points
    scatter = ax[0].scatter(x_train_transformed[:, 0], x_train_transformed[:, 1], c=y_train, edgecolor='black', s=20, linewidth=0.2)

    # Create legend
    legend_labels = {0: 'Group 0', 1: 'Group 1', 2: 'Group 2'}
    ax[0].legend(handles=scatter.legend_elements()[0], title='Classes', labels=[legend_labels[0], legend_labels[1], legend_labels[2]])
    ax[0].set_title("Decision surface of DTC (%d)" % depth)

    # Plot the decision tree
    plot_tree(dtc, ax=ax[1], filled=True, feature_names=["dayofweek", "hourofday"])

    plt.tight_layout()
    plt.show()
