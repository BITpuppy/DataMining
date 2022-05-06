import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import calendar
import seaborn as sns


# 绘制图像分析销售量随日期的变化趋势
def sales_date(dataset):
    day = [0 for i in range(len(dataset))]
    sales = [0 for i in range(len(dataset))]
    for i in range(len(dataset)):
        day[i] = pd.to_datetime(dataset.loc[i, 'date'])
        sales[i] = dataset.loc[i, 'transactions']
        
    plt.figure(figsize = (20, 10))
    plt.title('date-sales')
    plt.plot(day, sales)
    plt.show()


# 绘制图像分析油价随日期的变化趋势
def oilprice_date(dataset):
    day = [0 for i in range(len(dataset))]
    oil = [0 for i in range(len(dataset))]
    for i in range(len(dataset)):
        day[i] = pd.to_datetime(dataset.loc[i, 'date'])
        oil[i] = dataset.loc[i, 'oil-price']
    
    plt.figure(figsize = (20, 10))
    plt.title('date-oilprice')
    plt.plot(day, oil)
    plt.show()


# 绘制折线图分析不同时间间隔下日均销售量的变化趋势
def sales_time_interval(dataset):
    dataset = dataset.drop('oil-price', axis = 1)
    dataset = dataset.drop('holiday-type', axis = 1)
    sales_week = dataset.copy()
    sales_month = dataset.copy()
    sales_year = dataset.copy()

    sales_week["transactions"] = sales_week["transactions"].rolling(window=7, center=True, min_periods=3 ).mean()
    sales_month["transactions"] = sales_month["transactions"].rolling(window=30, center=True, min_periods=15).mean()
    sales_year["transactions"] = sales_year["transactions"].rolling(window=365, center=True, min_periods=183).mean()

    day = [0 for i in range(len(dataset))]
    week = [0 for i in range(len(dataset))]
    month = [0 for i in range(len(dataset))]
    year = [0 for i in range(len(dataset))]

    for i in range(len(dataset)):
        day[i] = pd.to_datetime(dataset.loc[i, 'date'])
        week[i] = sales_week.loc[i, 'transactions']
        month[i] = sales_month.loc[i, 'transactions']
        year[i] = sales_year.loc[i, 'transactions']

    plt.figure(figsize = (20, 10))
    plt.plot(day, week, c = 'b')
    plt.plot(day, month, c = 'g')
    plt.plot(day, year, c = 'r')
    plt.yticks([0, 30000, 60000, 90000, 120000, 150000])
    plt.legend(["7-day Average Sales", "30-day Average Sales", "365-day Average Sales"])
    plt.show()


# 绘制柱状图和饼图分析不同月份和季度的平均销售额
def sales_month_quarter(dataset):
    dataset.date = pd.to_datetime(dataset.date)
    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    dataset['quarter'] = dataset['date'].dt.quarter

    sales_month = dataset.groupby('month').agg({"transactions" : "mean"}).reset_index()
    sales_month['month'] = sales_month['month'].apply(lambda x: calendar.month_abbr[x])
    sales_quarter = dataset.groupby('quarter').agg({"transactions" : "mean"}).reset_index()

    plt.figure(figsize = (20, 10))
    plt.subplot(1, 2, 1)
    plt.barh(y = sales_month['month'], width = sales_month['transactions'])
    plt.title("Sales of different months in a year")

    plt.subplot(1, 2, 2)
    plt.pie(x = sales_quarter['transactions'], labels = sales_quarter['quarter'], autopct='%1.1f%%')
    plt.title("Sales of different quarters in a year")
    plt.show()


# 绘制柱状图比较不同商品种类的销售量
def sales_product_family(dataset):
    dataset = dataset[["family", "sales"]]
    dataset1 = dataset["family"].drop_duplicates()
    dataset2 = dataset.groupby(dataset["family"])["sales"].sum()

    size = len(dataset1)
    family = ["" for i in range(size)]
    sales = [0 for i in range(size)]
    for i in range(size):
        family[i] = dataset1[i]
        sales[i] = dataset2[i]

    plt.figure(figsize=(20, 20))
    plt.barh(y = family, width = sales)
    plt.title("Sales of different product family")
    plt.show()


# 绘制热力图分析不同商店之间的相关性
def correlation_among_shops(dataset):
    dataset = dataset[["store_nbr", "sales"]]
    dataset["ind"] = 1
    dataset["ind"] = dataset.groupby("store_nbr").ind.cumsum().values
    dataset = pd.pivot(dataset, index = "ind", columns = "store_nbr", values = "sales").corr()
    mask = np.triu(dataset)

    plt.figure(figsize=(20, 20))
    sns.heatmap(dataset, annot=True, fmt='.1f', cmap='bwr', square=True, mask=mask, linewidths=1, cbar=False)
    plt.title("Heatmap of stores")
    plt.show()


if __name__ == '__main__':
    dataset = pd.read_csv('dataset/data.csv')
    trainset = pd.read_csv('dataset/train.csv')

    print("Sales trend with date:")
    sales_date(dataset)
    print("Oil-price trend with date:")
    oilprice_date(dataset)
    print("Average sales with different time intervals:")
    sales_time_interval(dataset)
    print("Average sales in different months and quarters:")
    sales_month_quarter(dataset)
    print("Total sales in different product families:")
    sales_product_family(trainset)
    print("Correlation of all the shops:")
    correlation_among_shops(trainset)