import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix
from sklearn.tree import plot_tree

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', True)

# Reading Publibike availability data
dfPubliBikeAvailability = pd.read_csv("data/bike-availability-All-Stations_hourly.csv", encoding='latin-1', sep=';')

# Preparing Data
dfPubliBikeAvailability = dfPubliBikeAvailability.rename(columns={"Abfragezeit": "timestamp"})
dfPubliBikeAvailability["timestamp"] = pd.to_datetime(dfPubliBikeAvailability["timestamp"])
dfPubliBikeAvailability["dayofweek"] = dfPubliBikeAvailability["timestamp"].dt.weekday
dfPubliBikeAvailability["hourofday"] = dfPubliBikeAvailability["timestamp"].dt.hour
# Create a continuous numerical representation for the x-axis
dfPubliBikeAvailability["continuous_time"] = dfPubliBikeAvailability['dayofweek'] * 24 + dfPubliBikeAvailability['hourofday']

# Assigning the availability into 3 Groups;  Group "0" --> Available bikes = 0-1; Group "1" --> Available bikes = 2-4; Group "2" --> 5 or more
dfPubliBikeAvailability['bike_availability'] = dfPubliBikeAvailability['Bike']
dfPubliBikeAvailability['bike_availability'] = [0 if (i < 2) else i for i in dfPubliBikeAvailability['bike_availability']]
dfPubliBikeAvailability['bike_availability'] = [1 if (1 < i < 5) else i for i in dfPubliBikeAvailability['bike_availability']]
dfPubliBikeAvailability['bike_availability'] = [2 if (i > 4) else i for i in dfPubliBikeAvailability['bike_availability']]

dfPubliBikeAvailability['e-bike_availability'] = dfPubliBikeAvailability['EBike']
dfPubliBikeAvailability['e-bike_availability'] = [0 if (i < 2) else i for i in dfPubliBikeAvailability['e-bike_availability']]
dfPubliBikeAvailability['e-bike_availability'] = [1 if (1 < i < 5) else i for i in dfPubliBikeAvailability['e-bike_availability']]
dfPubliBikeAvailability['e-bike_availability'] = [2 if (i > 4) else i for i in dfPubliBikeAvailability['e-bike_availability']]

# Filter the DataFrame based on the specified station ID and desired date range
start_date = '2023-05-15'
end_date = '2023-09-15'
filtered_df = dfPubliBikeAvailability[(dfPubliBikeAvailability['timestamp'] >= start_date) &
                                      (dfPubliBikeAvailability['timestamp'] <= end_date) &
                                      (dfPubliBikeAvailability['id'] == 315)]

# With a dataframe with columns 'x', and 'y'
x = filtered_df[["continuous_time"]]
y = filtered_df['e-bike_availability']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Gradient Boosting Classifier
for n_est in (1, 4, 50):
    # Initialize GradientBoostingClassifier
    gbc = GradientBoostingClassifier(n_estimators=n_est, learning_rate=0.1, max_depth=3, random_state=42)

    # Fit the model
    gbc.fit(x_train, y_train)

    # Print the training scores
    print("Training score : %.3f (n_est=%d)" % (gbc.score(x_train, y_train), n_est))

    # Plot the decision tree
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_tree(gbc.estimators_[0][0], filled=True, feature_names=["continuous_time"], ax=ax)
    plt.title("Decision Tree Visualization (n_est=%d)" % n_est)
    plt.show()

    # Use the GradientBoostingClassifier to predict on the test data
    predictions = gbc.predict(x_test)

    # Flatten predictions and ensure it has the same length as y_test
    flattened_predictions = predictions.ravel()[:len(y_test)]

    # Evaluate the model using mean squared error, accuracy, precision, recall, F1-score, and Confusion Matrix and print results
    gbc_mse = mean_squared_error(y_test, flattened_predictions)
    accuracy = accuracy_score(y_test, flattened_predictions)
    precision = precision_score(y_test, flattened_predictions, average='weighted')
    recall = recall_score(y_test, flattened_predictions, average='weighted')
    f1 = f1_score(y_test, flattened_predictions, average='weighted')
    cm = confusion_matrix(y_test, flattened_predictions)

    print(f'Gradient Boosting Classifier Mean Squared Error: {gbc_mse}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')
    print('Confusion Matrix:')
    print(cm)
