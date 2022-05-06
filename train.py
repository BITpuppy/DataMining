import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fbprophet import Prophet      # 导入Prophet模型
import statsmodels.api as sm       # 导入ARIMA模型


# 使用MAE和MSE评估预测结果，并绘制图像比较预测值与真实值的误差
def Evaluation(y_true, y_pred):
    MAE = mean_absolute_error(y_true, y_pred)
    MSE = mean_squared_error(y_true, y_pred)
    print('MAE: %.3f' % MAE)
    print('MSE: %.3f' % MSE)

    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.show()


# Prophet模型
def Train_Prophet(dataset):
    # 数据集划分
    dataset.date = pd.to_datetime(dataset.date)
    train = dataset[dataset['date'] < dt.datetime(2017,1,1)][['date','sales']].groupby('date').mean().reset_index('date')
    train.columns = ['ds', 'y']

    test = dataset[dataset['date'] >= dt.datetime(2017,1,1)][['date','sales']].groupby('date').mean().reset_index('date')
    test.columns = ['ds', 'y']
    x_test = test['ds']
    x_test = x_test.to_frame()
    x_test.columns = ['ds']
    x_test['ds'] = pd.to_datetime(x_test['ds'])

    # 训练模型
    model = Prophet()
    model.fit(train)
    forecast = model.predict(x_test)

    model.plot(forecast)
    plt.show()

    y_pred = forecast['yhat'].values
    y_true = test['y'].values
    return y_true, y_pred


# ARIMA模型
def Train_ARIMA(dataset):
    # 数据集划分
    dataset.date = pd.to_datetime(dataset.date)
    train = dataset[['date','sales']].groupby('date').mean().reset_index('date')
    train_series = train['sales']
    train_series=pd.Series(train_series)
    train_series.index = pd.Index(train['date'])

    test = dataset[dataset['date'] >= dt.datetime(2017,1,1)][['date','sales']].groupby('date').mean().reset_index('date')

    # 训练模型
    model = sm.tsa.arima.ARIMA(train_series, order = (150, 1, 0)).fit()
    forecast = model.predict('2017-01-01', '2017-08-15', dynamic=True)

    print(forecast)
    
    y_pred = forecast.values
    y_true = test['sales'].values
    return y_true, y_pred


if __name__ == '__main__':
    dataset = pd.read_csv('dataset/train.csv')
    
    print("Training Prophet Model......")
    y_true1, y_pred1 = Train_Prophet(dataset)
    print("Evaluating Prophet Model......")
    Evaluation(y_true1, y_pred1)

    print("Training ARIMA Model......")
    y_true2, y_pred2 = Train_ARIMA(dataset)
    print("Evaluating ARIMA Model......")
    Evaluation(y_true2, y_pred2)