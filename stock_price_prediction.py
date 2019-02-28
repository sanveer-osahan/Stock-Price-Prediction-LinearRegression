import pandas as pd
import math
import datetime
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

# read data from csv as dataframe and set Date column as index
df = pd.read_csv("stock_price_data.csv")
df.set_index('Date', inplace=True)

# setting 4 features as
# 1. Close
# 2. High-Low percentage
# 3. Percent change from Open to Close
# 4. Volume
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# Output for each row in the dataframe will be Close price of 1% of the total data time in future
# For eg:-
# if the dataframe has 1000 rows,
# output for each row will be Close price of future date which is 10 days after the current date
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))  # 0.01 denotes 1%

# Shifting the Close price columns by 1% upwards as label column for rows in dataframe
df['label'] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['label'], 1))  # Setting columns except label as input features
X = preprocessing.scale(X)  # normalizing the input features for eliminating sparsity
X_lately = X[-forecast_out:]  # data for which output has to be predicted
X = X[:-forecast_out]  # data used for training and testing

df.dropna(inplace=True)
Y = np.array(df['label'])  # Setting label column as output

# splitting the data - 80% for training and 20% for testing the model
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

# Training the classifier with Linear Regression model
# Comment below lines to skip traning and get classifier from pickle file
clf = LinearRegression()
clf.fit(X_train, Y_train)

with open('stock_price_classifier.pickle', 'wb') as f:
    pickle.dump(clf, f)

# Uncomment to reuse the classifier dumped to pickle file to skip training every time program is executed
'''
pickle_in = open('stock_price_classifier.pickle', 'rb')
clf = pickle.load(pickle_in)'''

accuracy = clf.score(X_test, Y_test)

# Make the actual prediction
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

# Plotting the data on a graph
df['Forecast'] = np.nan

last_date = datetime.datetime.strptime(df.iloc[-1].name, '%Y-%m-%d')
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
