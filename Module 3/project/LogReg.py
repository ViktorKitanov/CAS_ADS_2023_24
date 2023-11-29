import numpy as np
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

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
x = filtered_df[["continuous_time"]]
y = filtered_df['bike_availability']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train logistic regression classifier
clf = LogisticRegression(solver='sag', max_iter=1000, multi_class='ovr')
clf.fit(x_train_scaled, y_train)

coef = clf.coef_
intercept = clf.intercept_
print(coef, intercept)

plt.scatter(x_test, y_test, marker='*', label='data points')

# Generate a range of x values
x_values = np.linspace(x_test.min(), x_test.max(), 300).reshape(-1, 1)

# Use the fitted model to get predicted probabilities for the range of x_values
probabilities = clf.predict_proba(x_values)[:, 1]

# Plot the logistic function
plt.plot(x_values, probabilities, label='Logistic Function', c='r')
plt.xlabel('x values')
plt.ylabel('Probability (y values)')
plt.legend()
plt.show()

# Trained logistic regression model, now make predictions
predictions = clf.predict(x_test)

# Evaluate the model using accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# Evaluate the model using precision, recall, and F1-score
precision = precision_score(y_test, predictions, average='micro')
recall = recall_score(y_test, predictions, average='micro')
f1 = f1_score(y_test, predictions, average='micro')

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

from sklearn.metrics import confusion_matrix

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
print('Confusion Matrix:')
print(cm)