# min-CVaR
Real-time risk management with PyPortfolioOpt min_cvar method using neural networks

## Data preprocessing

```python
import pandas as pd

# load the daily OHLCV prices of 4 stocks
pko = pd.read_csv(path_to_pko_d.csv)
dnp = pd.read_csv(path_to_dnp_d.csv)
pgn = pd.read_csv(path_to_pgn_d.csv)
pkn = pd.read_csv(path_to_pkn_d.csv)

# prepare labels to drop
label = ['Otwarcie',
         'Najwyzszy',
         'Najnizszy',
         'Wolumen']

# drop labels
pko = pko.drop(label, axis = 1)
dnp = dnp.drop(label, axis = 1)
pgn = pgn.drop(label, axis = 1)
pkn = pkn.drop(label, axis = 1)

# add the rest of stocks closing prices to dnp variable. 
dnp["PGN"] = pgn["Zamkniecie"]

# change the datetime to mutual timespan across 4 stocks
dnp = dnp[dnp['Data'] >= '2017-11-10']
dnp = dnp.reset_index(drop=True)
dnp["PKO"] = pko["Zamkniecie"]
dnp["PKN"] = pkn["Zamkniecie"]

# rename columns 
dnp = dnp.rename(columns={"Zamkniecie": "DNP", "Data": "Date"})

# set the index to first column "Date"
asset_price = dnp.set_index(dnp.columns[0])

# check data by printing preprocessed stock prices
print(asset_price)

# save preprocessed stock prices to csv called "asset_price"
asset_price.to_csv("asset_price.csv")
```

## Portfolio optimization
To deal with optimization problem we will use min_cvar() method provided by the EfficientCVaR class which is a part of PyPortfolioOpt library, see documentation [here](https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html)

```python
import pandas as pd

# load the portoflio dataset
df = pd.read_csv(path_to_asset_price.csv, parse_dates=True, index_col="Date")

print(df.shape)

# set a number which will split the dataset, in this example we want to split data into a 7-day window of asset prices
n_split = 166

import numpy as np

# create dictionary containing keys for every epoch in splitted dataset
df_split = np.array_split(df, n_split)
n_keys = np.arange(n_split)
prices_dict = dict(zip(n_keys, df_split))

mean_dict = {}
mu_dict = {}
returns_dict = {}
ef_dict = {}
min_cvar_weights_dict = {}

from pypfopt import expected_returns
from pypfopt import EfficientCVaR

# loop through epochs to create minimum CVaR portfolio weights corresponding to each epoch
for x in prices_dict.keys():
    sub_prices = prices_dict[x]
    sub_returns = sub_prices.pct_change()
    mean_dict[x] = np.mean(sub_returns)
    mu_dict[x] = expected_returns.mean_historical_return(sub_prices)
    returns_dict[x] = expected_returns.returns_from_prices(sub_prices)
    ef_dict[x] = EfficientCVaR(mu_dict[x], returns_dict[x])
    min_cvar_weights_dict[x] = ef_dict[x].min_cvar(market_neutral=False)
    
weights_dict = {}

lists = [[] for i in range(n_split)]

# loop through each Ordered Dictionary to extract asset weights and save them as dictionary called "weights_dict" 
for x in min_cvar_weights_dict.keys():
    sub_list = lists[x]
    sub_values = min_cvar_weights_dict[x]
    for key, value in sub_values.items():
        sub_list.append(value)
        weights_dict[x] = sub_list
```

## Weights predictions with neural network

```python
# create input and output datasets for training a neural network 
df_returns_raw = pd.DataFrame(mean_dict)
df_returns = df_returns_raw.transpose()

labels = ['DNP',
          'PGN', 
          'PKO', 
          'PKN']

df_weights_raw = pd.DataFrame(weights_dict, index=labels)
df_weights = df_weights_raw.transpose()

from sklearn.model_selection import train_test_split

# split the dataset to create training and test sets
x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(df_returns, df_weights, test_size=.3, random_state=123)

# convert training and test sets to Numpy matrix for fitting the model
x_train = np.matrix(x_train_raw)
x_test = np.matrix(x_test_raw)
y_train = np.matrix(y_train_raw)
y_test = np.matrix(y_test_raw)

from keras.layers import Dense
from keras.models import Sequential

n_cols = x_train.shape[1]

# create a Sequential neural network with two hidden layers, one input layer and one output layer using the ReLU activation
model = Sequential() 
model.add(Dense(64, input_dim=n_cols, activation='relu')) 
model.add(Dense(32, activation = 'relu')) 
model.add(Dense(4, activation = 'relu'))

# compile the model setting aptimizer to "Adam" and loss function to "Mean Squared Error"
model.compile(optimizer='adam', loss='mse')

# fit the model to the training set 
model.fit(x_train, y_train, epochs=10)

from sklearn.metrics import r2_score

# obtain predictions from model on the test set
predictions = model.predict(x_test)

# calculate the R^2 score on the test set
print("R^2 Score : ", r2_score(y_test, predictions))

# use the model to predict what the minimum CVaR portfolio would be, when new asset data is presented
assets_returns = np.array([0.001030, 0.004332, 0.000126, -0.002447]) 
assets_returns.shape = (1,4) 

print("Predicted weights for minimum CVaR portfolio : ", model.predict(assets_returns))
```
