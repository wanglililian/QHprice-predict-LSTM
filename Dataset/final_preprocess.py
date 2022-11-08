import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 这个函数就是用于设置pd读取的数据print输出的格式，可以不用看
def setup_pandas():
    # 显示所有列
    pd.set_option('display.max_columns', 200)

    # 显示所有行
    pd.set_option('display.max_rows', None)

    # 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 1000)
    pd.set_option('expand_frame_repr', False)


# 绘图函数。使用该图绘制了螺纹钢close的变化折线图
def display(df, name, use_data_ratio=0.01):
    import seaborn as sns

    sns.set_theme(style="darkgrid")  # 设置图像的格式
    plt.figure(figsize=(20, 5))  # 定义figure画板，并设置大小
    plt.legend([name])  # 给图像加图例
    sns.lineplot(x="Date", y=name, data=df[:int(use_data_ratio * len(df))])  # 以天为单位画图，横轴x=date日期；纵轴y=name=close；data控制绘制图像的数据的范围
    plt.title('Historical daily data of luowengang', fontsize=14)  # 图像标题
    plt.ylabel(name)  # 设置纵轴的名
    plt.show()  # 显示图像


def preprocess(read_path, save_path, minmax=True):
    # 1、读取数据
    df = pd.read_excel(
        read_path,
        usecols=['日期', '收盘', '开盘', '最高', '最低']
    )

    df.columns = ['Date', 'Close', 'Open', 'High', 'Low']    # 设置数据的列名称
    df.Date = pd.to_datetime(df.Date)  # 设置data列的格式为pandas中的日期格式
    df = df.sort_values(by=['Date'])  # 按照日期对数据进行排序
    df = df.set_index('Date', inplace=False).dropna()  # 设置数据index为date
    print(df.columns)


    # 3、对数据进行一些处理，计算后续模型需要的数据
    # 3.1 将日期数据进行分解，为年、月、日、工作日，分别进行存储
    df['Date'] = df.index
    df['Year'], df['Month'], df['Week'], df['Weekday'], df[
        'Day'] = df.index.year, df.index.month, df.index.isocalendar().week, df.index.weekday, df.index.day

    # 3.2 计算returns，涨跌幅=（现价—上一个交易日结算价）/上一交易日结算价×100%。diff()作差函数；shift(1)表示所有数据上移一格
    df['Returns'] = df['Close'].diff() / (df['Close'].shift(1))
    # 将数据中的Nan删除
    df = df.dropna()

    # 3.3 滚动计算周平均以及月平均close值
    # pre表示当前天前一周
    # next表示当前天所在的周
    # 1,2,4分别为一周，两周，一个月
    for week_num in [1, 2, 4]:
        df['Pre_{}_Week_Mean'.format(week_num)] = df['Close'].rolling(window=week_num * 5).mean().shift(periods=1)
        df['Next_{}_Week_Mean'.format(week_num)] = df['Close'].rolling(window=week_num * 5).mean().shift(periods=-week_num * 5 + 1)
        df['Next_{}_Week_Mean_Diff'.format(week_num)] = df['Next_{}_Week_Mean'.format(week_num)] - df['Pre_{}_Week_Mean'.format(week_num)]
        df['Next_{}_Week_Mean_Returns'.format(week_num)] = df['Next_{}_Week_Mean_Diff'.format(week_num)] / df['Pre_{}_Week_Mean'.format(week_num)]

    # 4、保存处理好的数据
    print('start to save csv')
    df.to_csv('../data/final_lwg_2009-2022_Day.csv')
    print('finish to save csv')

if __name__ == '__main__':
    setup_pandas()

    preprocess(
        read_path='../data/螺纹钢主力.xlsx',
        save_path='../data/final_lwg_2009-2022_Day_minmax.csv',
        minmax=True
    )
