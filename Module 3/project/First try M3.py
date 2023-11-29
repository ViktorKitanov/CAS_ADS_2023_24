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

print(filtered_df)

# With a dataframe with columns 'x', and 'y'
x = filtered_df[["continuous_time"]]
y = filtered_df['bike_availability']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#DECISION TREES AND RANDOM FORESTS
#CG

# Decision Tree Model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(x_train, y_train)
dt_predictions = dt_model.predict(x_test)

# Random Forest Model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(x_train, y_train)
rf_predictions = rf_model.predict(x_test)

# Evaluate the models
dt_mse = mean_squared_error(y_test, dt_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)

print(f'Decision Tree Mean Squared Error: {dt_mse}')
print(f'Random Forest Mean Squared Error: {rf_mse}')

# Plotting the results
plt.figure(figsize=(12, 6))

# Scatter plot for actual vs. predicted values for Decision Tree
plt.subplot(1, 2, 1)
plt.scatter(y_test, dt_predictions, alpha=0.5)
plt.title('Decision Tree: Actual vs. Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Scatter plot for actual vs. predicted values for Random Forest
plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_predictions, alpha=0.5)
plt.title('Random Forest: Actual vs. Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()

# Scatter plot for Decision Tree
plt.subplot(1, 2, 1)
sns.scatterplot(x=x_test['continuous_time'], y=y_test, hue=dt_predictions, palette='viridis', alpha=0.8)
plt.title('Decision Tree: Bike Availability Groups with Trendline')
plt.xlabel('Continuous Time')
plt.ylabel('Bike Availability Group')

# Scatter plot for Random Forest
plt.subplot(1, 2, 2)
sns.scatterplot(x=x_test['continuous_time'], y=y_test, hue=rf_predictions, palette='viridis', alpha=0.8)
plt.title('Random Forest: Bike Availability Groups with Trendline')
plt.xlabel('Continuous Time')
plt.ylabel('Bike Availability Group')

plt.tight_layout()
plt.show()

#CS


# Create a DataFrame with the generated data
#df = pd.DataFrame({'continuous_time': x['continuous_time'].tolist(), 'bike_availability': y.values}, index=filtered_df.index)





#scaler = MinMaxScaler()
#df[['continuous_time']] = scaler.fit_transform(df[['continuous_time']])

#n_d = 2

# Define the model
#model = tf.keras.models.Sequential([
#    tf.keras.layers.Flatten(input_shape=(n_d,)),  # Adjust input shape based on the number of features
#    tf.keras.layers.Dense(10000, activation=tf.keras.layers.LeakyReLU()),
#    tf.keras.layers.Dense(500, activation=tf.keras.layers.LeakyReLU()),
#    tf.keras.layers.Dense(2, activation='linear'),  # Use linear activation for regression problems
#])

#from keras.models import Sequential
#from keras.layers import Dense

#model = Sequential()

# Input layer
#model.add(Dense(units=64, input_shape=(1,), activation='relu'))

# Hidden layers (experiment with the number of hidden layers and units)
#model.add(Dense(units=128, activation='relu'))
#model.add(Dense(units=64, activation='relu'))

# Output layer
#model.add(Dense(units=1, activation='softmax'))

# Compile the model
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Compile the model
#model.compile(optimizer='adam', loss='mean_squared_error')  # Use appropriate loss for regression problems

# Train the model with your dataframe
#X_train = df[['continuous_time']].values
#y_train = df['bike'].values
#model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict on the same data
#y_pred = model.predict(X_train)

# Visualize the results
#fig, ax = plt.subplots(figsize=(8, 6))
#ax.plot(df['continuous_time'], df['bike'], 'bx', label='Actual')
#ax.plot(df['continuous_time'], y_pred, 'ro', label='Predicted')
#ax.set_xlabel('continuous_time')
#ax.set_ylabel('bike')
#ax.legend()
#plt.show()

# Create a continuous numerical representation for the x-axis
#filtered_df.loc[:,'continuous_time'] = filtered_df['dayofweek'] * 24 + filtered_df['hourofday']

# Create a violin plot
#fig = px.violin(filtered_df, x='continuous_time', y='e-bike_availability',
#                labels={'continuous_time': 'Continuous Time', 'e-bike_availability': 'Bike Availability'},
#                title='Violin Plot of Bike Availability over Continuous Time')

# Show the plot
#fig.show()