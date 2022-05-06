import pandas as pd

if __name__ == '__main__':
    # 读入数据集
    df_train = pd.read_csv("dataset/train.csv", encoding="utf-8")
    df_test = pd.read_csv("dataset/test.csv")
    df_transactions = pd.read_csv("dataset/transactions.csv")
    df_oil = pd.read_csv("dataset/oil.csv")
    df_holiday = pd.read_csv("dataset/holidays_events.csv")
    df_stores = pd.read_csv("dataset/stores.csv")

    table1 = df_transactions
    table2 = df_oil
    table3 = df_holiday

    # 将每日的全部交易量累加
    table1 = table1.drop("store_nbr", axis = 1)
    table1 = table1["transactions"].groupby(table1["date"]).sum()
    table1 = table1.to_frame()

    # 将两个数据表格按照日期信息合并
    table12 = pd.merge(table1, table2, how='left', on='date')

    table3 = table3.drop("locale", axis = 1)
    table3 = table3.drop("locale_name", axis = 1)
    table3 = table3.drop("description", axis = 1)
    table3 = table3.drop("transferred", axis = 1)
    table = pd.merge(table12, table3, how='left', on=['date', 'date'])

    # 填补缺失值
    table['dcoilwtico'] = table['dcoilwtico'].fillna(method='bfill')
    table['type'] = table['type'].fillna('Normal')
    
    # 对数据的列进行重命名
    table.columns = ["date", "transactions", "oil-price", "holiday-type"]
    table.to_csv("data.csv", index=0, sep=',')