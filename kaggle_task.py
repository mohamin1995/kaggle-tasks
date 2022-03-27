import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import VotingRegressor


def rmsle(predicted, real):
    sum = 0.0
    for x in range(len(predicted)):
        if predicted[x] > 0:
            p = np.log(predicted[x] + 1)
            r = np.log(real[x] + 1)
            sum = sum + (p - r) ** 2
    return (sum / len(predicted)) ** 0.5


stores = pd.read_csv('D:/kaggle/stores.csv')
oil = pd.read_csv('D:/kaggle/oil.csv')
transactions = pd.read_csv('D:/kaggle/transactions.csv')
train = pd.read_csv('D:/kaggle/train.csv')
holidays_events = pd.read_csv('D:/kaggle/holidays_events.csv')

idx = pd.date_range('01-01-2013', '08-31-2017')
all_days_df = pd.DataFrame({'date': idx.to_series()})
all_days_df['date'] = all_days_df.date.astype(str)
oil = pd.merge(all_days_df, oil, on=['date'], how='left')
oil.dcoilwtico = oil.dcoilwtico.interpolate()

holidays_events.loc[holidays_events.type == 'Additional', 'type'] = 'Holiday'
holidays_events.loc[holidays_events.type == 'Transfer', 'type'] = 'Holiday'
holidays_events.loc[holidays_events.type == 'Bridge', 'type'] = 'Holiday'

holidays_events = holidays_events[(holidays_events.locale == 'National') &
                                  (holidays_events.type == 'Holiday') & (holidays_events.transferred == False)]
holidays_events = holidays_events[['date', 'type']]
holidays_events = holidays_events.rename(columns={"type": "holiday"})

train = pd.merge(train, stores, how='left')
train = pd.merge(train, oil, how='left')
train = pd.merge(train, transactions, on=['date', 'store_nbr'], how='left')
train = pd.merge(train, holidays_events, on=['date'], how='left')

train['holiday'] = np.where(train['holiday'] == 'Holiday', True, False)

ord_enc = OrdinalEncoder()
family_enc = LabelBinarizer(sparse_output=False)
city_enc = LabelBinarizer(sparse_output=False)
state_enc = LabelBinarizer(sparse_output=False)
cluster_enc = LabelBinarizer(sparse_output=False)

train["type"] = ord_enc.fit_transform(train[["type"]])

oh_family = family_enc.fit_transform(train['family'])
oh_city = city_enc.fit_transform(train['city'])
oh_state = state_enc.fit_transform(train['state'])
oh_cluster = cluster_enc.fit_transform(train['cluster'])

train = pd.concat(
    [train, pd.DataFrame(oh_family, columns=['family_' + str(i) for i in range(1, len(oh_family[0]) + 1)]),
     pd.DataFrame(oh_city, columns=['city_' + str(i) for i in range(1, len(oh_city[0]) + 1)]),
     pd.DataFrame(oh_state, columns=['state_' + str(i) for i in range(1, len(oh_state[0]) + 1)]),
     pd.DataFrame(oh_cluster, columns=['cluster_' + str(i) for i in range(1, len(oh_cluster[0]) + 1)])], axis=1)

train = train.sort_values(by='date')

train['dcoilwtico'] = train['dcoilwtico'].fillna(method='bfill')
train['transactions'] = train['transactions'].fillna(train.transactions.median())

scaler = MinMaxScaler()

train['date'] = pd.to_datetime(train['date'])
train['dow'] = train.date.dt.dayofweek
train[['onpromotion', 'dcoilwtico', 'transactions', 'dow']] = scaler.fit_transform(
    train[['onpromotion', 'dcoilwtico', 'transactions', 'dow']])

train['high_wages'] = False
train['after_earthquake'] = False

train.loc[((train.date.dt.day >= 15) & (train.date.dt.day <= 20)) | (
            (train.date.dt.day >= 1) & (train.date.dt.day <= 5)), 'high_wages'] = True
train.loc[((train.date.dt.day >= 15) & (train.date.dt.day <= 20)) | (
            (train.date.dt.day >= 1) & (train.date.dt.day <= 5)), 'high_wages'] = True

train.loc[(train.date >= '2016-04-16') & (train.date <= '2016-05-16'), 'after_earthquake'] = True

train = train.drop(['id', 'store_nbr', 'date', 'family', 'city', 'state', 'cluster'], 1)

x = train.loc[:, train.columns != 'sales']
y = train.loc[:, 'sales']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

reg1 = tree.DecisionTreeRegressor(max_depth=40, random_state=11)
reg2 = tree.DecisionTreeRegressor(max_depth=40, random_state=5)

reg = VotingRegressor(estimators=[('one', reg1), ('two', reg2)])


reg = reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))
print('rmsle is {}'.format(rmsle(y_pred, y_test.to_list())))

test_data = pd.read_csv('D:/kaggle/test.csv')
test_data = pd.merge(test_data, stores, how='left')
test_data = pd.merge(test_data, oil, how='left')
test_data = pd.merge(test_data, transactions, on=['date', 'store_nbr'], how='left')
test_data = pd.merge(test_data, holidays_events, on=['date'], how='left')

test_data['date'] = pd.to_datetime(test_data['date'])

test_data['holiday'] = np.where(test_data['holiday'] == 'Holiday', True, False)
test_data["type"] = ord_enc.transform(test_data[["type"]])

oh_family = family_enc.transform(test_data['family'])
oh_city = city_enc.transform(test_data['city'])
oh_state = state_enc.transform(test_data['state'])
oh_cluster = cluster_enc.transform(test_data['cluster'])

test_data = pd.concat(
    [test_data, pd.DataFrame(oh_family, columns=['family_' + str(i) for i in range(1, len(oh_family[0]) + 1)]),
     pd.DataFrame(oh_city, columns=['city_' + str(i) for i in range(1, len(oh_city[0]) + 1)]),
     pd.DataFrame(oh_state, columns=['state_' + str(i) for i in range(1, len(oh_state[0]) + 1)]),
     pd.DataFrame(oh_cluster, columns=['cluster_' + str(i) for i in range(1, len(oh_cluster[0]) + 1)])], axis=1)

test_data = test_data.sort_values(by='date')

test_data['dcoilwtico'] = test_data['dcoilwtico'].fillna(method='bfill')

aggregated_trans = pd.DataFrame(transactions.groupby(['store_nbr']).transactions.median())
test_data = pd.merge(test_data, aggregated_trans, on=['store_nbr'], how='left')
test_data['transactions_x'] = test_data['transactions_x'].fillna(test_data['transactions_y'])

test_data['dow'] = test_data.date.dt.dayofweek

test_data['high_wages'] = False
test_data['after_earthquake'] = False

test_data.loc[((test_data.date.dt.day >= 15) & (test_data.date.dt.day <= 20)) | (
            (test_data.date.dt.day >= 1) & (test_data.date.dt.day <= 5)), 'high_wages'] = True
test_data.loc[((test_data.date.dt.day >= 15) & (test_data.date.dt.day <= 20)) | (
            (test_data.date.dt.day >= 1) & (test_data.date.dt.day <= 5)), 'high_wages'] = True

test_data.loc[(test_data.date >= '2016-04-16') & (test_data.date <= '2016-05-16'), 'after_earthquake'] = True

test_data[['onpromotion', 'dcoilwtico', 'transactions_x', 'dow']] = scaler.transform(
    test_data[['onpromotion', 'dcoilwtico', 'transactions_x', 'dow']])
test_data = test_data.drop(
    ['store_nbr', 'date', 'family', 'city', 'state', 'cluster', 'transactions_y'], 1)
test_data.head()

result = reg.predict(test_data.loc[:, test_data.columns != 'id'])
sales = pd.Series(result, name='sales')
test_data = pd.concat([test_data, sales], axis=1)
test_data.to_csv('D:/kaggle/output_with_details.csv')
test_data.to_csv('D:/kaggle/output.csv', columns=['id', 'sales'], index=False)
