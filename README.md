# QHprice-predict-LSTM
## 代码各文件介绍
code文件夹内是模型代码：

- checkpoints  存储训练好的模型参数文件，以供后续test时加载模型  
- data	数据集文件，测试test模式的输出csv也会存储在该路径下  
- Dataset   数据预处理相关文件  
  - CLDataset.py  重写Dataset类，重写了__getitem__()函数以供后续数据加载器DataLoader使用，①用于划分数据集，train、test、val；②并且该文件内部选择相关列作为特征x以及y，z  
  - final_preprocess.py  对爬取的螺纹钢数据进行预处理，①对数据进行分解；②计算相关数据（pre_week_mean等）以供后续选择特征；③并且本代码预测的未来的周平均数据，所以也要提前计算周平均作为标签数据  
- models  该文件夹存储模型定义相关文件   
  - GRU.py	定义了GRU的网络结构，重写了forward前向传播函数  
- solvers	该文件夹中保存了模型的相关solver文件，solver类用于将模型的定义、训练、测试等函数封装
  - BasicSolver.py	  定义了一个solver基类
  - GRUSolver.py	  定义了一个GRU模型的basicSolver子类。重写了build_model()函数以及run_one_batch()函数，一个是构建模型的函数一个是训练一轮的函数
- tesorboard_logs	该文件夹就存储模型的可视化数据以及训练的输出结果，相关命令见前面
- utils	该文件夹中定义了一些工具函数
  - automatic_weighted_loss.py		可以不看，就是定义了一种loss自动加权类，本模型忽略
  - metric.py		定义了一些metric的计算函数，包括MAE，MSE、MAPE，MEDIAN_MAPE等，（相关计算以及含义自己百度吧）可以写在论文里，本模型主要是使用了这几个，后面还会介绍
  - plot_results.py   定义了画图函数，在basicsolver中调用该函数画了pred，true的图，具体图像保存在了tensorboard中
  - setup_seed.py    定义了初始化种子的函数，可以不管，定义了这个可以保证模型训练初始化是一样的，具体可以自己百度
  - utils.py		定义了一些工具函数，包括①EarlyStopping提前停止训练的类，防止过拟合，文件中有代码注释；②adjust_learning_rate 调整学习率函数，在训练的过程中要逐渐衰减学习率。

数据爬取文件夹内是螺纹钢数据爬取相关代码，使用jupyter notebook打开。

## 模型介绍
采用GRU模型预测未来的期货价格，本模型拟实现预测未来一周的平均价格、未来两周的平均价格、未来一个月的平均价格

模型结构如下所示：
![image](https://user-images.githubusercontent.com/52337721/200462570-865b66c3-bc3c-41ec-8367-0bb9566e5942.png)


## 数据预处理
首先需要进行数据预处理：
已有数据：
'日期', '收盘价', '开盘价', '最高价', '最低价', '涨跌幅收盘价', '振幅'
'Date', 'Close', 'Open', 'High', 'Low', 'Returns', 'Amplitude'

returns：涨跌幅=（现价—上一个交易日结算价）/上一交易日结算价×100%

1、特征工程
- Pre_1_Week_Mean：这一天前5天（一周）的close的均值。pre表示之前。
- Next_1_Week_Mean：这一天为首的一周的close的均值。next表示之后。
- Next_1_Week_Mean_Diff：这一天为首的一周的close均值的差值（与前一周的差值）  某天的Next_1_Week_Mean_Diff=Next_1_Week_Mean-Pre_1_Week_Mean。
- Next_1_Week_Mean_Returns：这一天为首的一周的close均值的涨跌幅。 某天的 Next_1_Week_Mean_Returns=Next_1_Week_Mean_Diff/Pre_1_Week_Mean

2、数据归一化
本代码对于数据归一化的处理在数据预处理时实现，（final_preprocess.py 文件内），采用的是minmax归一化
# 
    if minmax:
        print('min = {}, max = {}'.format(df['Close'].min(), df['Close'].max()))
        df = (df - df.min()) / (df.max() - df.min())

## 数据集的划分
原始数据：2009-2022年，3000+条

- train：2009-2018 
- test： 2018-2021 
- val：  2021-2022 

# 训练数据的加载
- x：特征数据；
- y：标签数据；
- z：用于保存模型预测结果时保存相关的日期等信息；

x_used_columns = ['Close', 'Open', 'Low', 'High']  
y_used_columns = ['Next_1_Week_Mean']    
z_used_columns = ['Pre_1_Week_Mean','Year', 'Month', 'Day', 'Weekday']  


## 模型损失函数
    def _select_criterion(self):
				if self.args.task == 'Regression':
            if self.args.use_mae_loss:
                criterion = nn.L1Loss()
            else:
                criterion = nn.MSELoss()
				……
本模型只是进行了regression回归任务，所以使用的损失函数包括两种：（并且使用参数args.use_mae_loss控制）
- L1Loss：
    - 平均绝对误差（MAE）是另一种用于回归模型的损失函数。MAE是目标变量和预测变量之间绝对差值之和。
    - 计算公式如下
    
![image](https://user-images.githubusercontent.com/52337721/200462091-48bbff8f-52f5-4224-ab67-9733f0f949cd.png)

- MSELoss：

![image](https://user-images.githubusercontent.com/52337721/200462119-0514be38-19e3-4a6a-8cbd-ba71275cb551.png)

