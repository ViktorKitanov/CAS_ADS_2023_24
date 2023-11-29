import numpy as np
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

# Train logistic regression classifier
clf = LogisticRegression(solver='sag', max_iter=100, multi_class='ovr')
clf.fit(x_train, y_train)

def plot_decision_surface(x_train, y_train, classifier):
    """
    Creates 2D mesh, predicts class for each point on the mesh, and visualizes it
    """

    x_train_min, x_train_max = x_train.iloc[:, 0].min() - 1, x_train.iloc[:, 0].max() + 1
    y_train_min, y_train_max = x_train.iloc[:, 1].min() - 1, x_train.iloc[:, 1].max() + 1

    mesh_step = 0.02
    xx, yy = np.meshgrid(np.arange(x_train_min, x_train_max, mesh_step),
                         np.arange(y_train_min, y_train_max, mesh_step))

    # Create mesh nodes for prediction
    mesh_nodes = np.c_[xx.ravel(), yy.ravel()]

    # Predict the class for each mesh node
    mesh_nodes_class = classifier.predict(mesh_nodes)

    # Reshape the predictions to the shape of the meshgrid
    mesh_nodes_class = mesh_nodes_class.reshape(xx.shape)

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the decision surface
    ax.contourf(xx, yy, mesh_nodes_class, cmap='viridis', alpha=0.5)

    # Scatter plot for training data points
    scatter = ax.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=y_train, cmap=plt.cm.Paired,
                         edgecolor='gray', s=30, linewidth=0.2)

    # Add colorbar for reference
    cbar = plt.colorbar(scatter, ticks=[0, 1, 2])
    cbar.set_label('Class')
    cbar.set_ticklabels(['Class 0', 'Class 1', 'Class 2'])

    ax.set_title("Decision surface of Logistic Regression")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day")

    plt.show()

# Call the function to plot the decision surface
plot_decision_surface(x_train, y_train, classifier=clf)
