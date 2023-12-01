import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier from sklearn.ensemble
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

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

# Assigning the availability into 3 Groups; Group "0" --> Available bikes = 0-1; Group "1" --> Available bikes = 2-4; Group "2" --> 5 or more
dfPubliBikeAvailability['bike_availability'] = pd.cut(dfPubliBikeAvailability['Bike'], bins=[-np.inf, 1, 4, np.inf], labels=[0, 1, 2])
dfPubliBikeAvailability['e-bike_availability'] = pd.cut(dfPubliBikeAvailability['EBike'], bins=[-np.inf, 1, 4, np.inf], labels=[0, 1, 2])

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

# RANDOM FOREST
x_train_transformed = x_train.values

# Gradient Boosting Classifier
for max_depth in (3, 5, 7):
    for n_est in (1, 50, 100):
        # Initialize GradientBoostingClassifier
        gbc = GradientBoostingClassifier(n_estimators=n_est, learning_rate=0.1, max_depth=max_depth, random_state=42)

        # Fit the model
        gbc.fit(x_train_transformed, y_train)

        # Print the training scores
        print("Training score : %.3f (n_est=%d)" % (gbc.score(x_train_transformed, y_train), n_est))

        fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=150)

        # Plot decision boundaries
        h = 0.02
        x_min, x_max = x_train_transformed[:, 0].min() - 1, x_train_transformed[:, 0].max() + 1
        y_min, y_max = y_train.min() - 0.1, y_train.max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Create a meshgrid with a single feature (continuous_time)
        meshgrid_data = np.c_[xx.ravel()]
        Z = gbc.predict(meshgrid_data)
        Z = Z.reshape(xx.shape)
        ax[0].contourf(xx, yy, Z, alpha=0.5)

        # Plot training points
        scatter = ax[0].scatter(x_train_transformed[:, 0], y_train, c=y_train, edgecolor='black', s=20, linewidth=0.2)

        # Create legend
        legend_labels = {0: 'Group 0', 1: 'Group 1', 2: 'Group 2'}
        ax[0].legend(handles=scatter.legend_elements()[0], title='Classes', labels=[legend_labels[0], legend_labels[1], legend_labels[2]])
        ax[0].set_title("Decision surface of GBC (n_est=%d)" % n_est)

        # Plot the decision tree
        for tree in gbc.estimators_:
            plot_tree(tree[0], ax=ax[1], filled=True, feature_names=["continuous_time"])
        plt.tight_layout()
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
