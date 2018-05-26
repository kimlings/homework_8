# 1. 请尝试用拟合回归的方法，预测2018年5.26日 的车牌价。
# 原始资料请见： http://jt.gz.bendibao.com/news/2015427/186209.shtml
# 2. 要求： 肯定不符合线性模式。其它模型需要大家自学一下sklearn。：）
# 3. 可以考虑分段，不一定要拟合所有数据
# 4. 请用matplotlib先画出自2012年8月到现在的价格变化趋势，拟合可以用大家想到的任何方法。但需要有分析。
# 5. 请分别预测个人最低成交价和个人平均成交价。

import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

# 1.读取数据
array_price = np.loadtxt('chepaijiage.csv', delimiter=",", skiprows=1, dtype=str)  # 读入数据，并存在数组array__price里面
row, col = array_price.shape
t = []  # 新建一个空的列表用来存时间
for i in range(row):
    t.append(datetime.datetime.strptime(array_price[i, 0], '%Y/%m/%d'))  # 把array_price数组里的日期字符串改为时间格式
personal_lowest_price = np.zeros(row)
personal_average_price = np.zeros(row)
for i in range(row):
    personal_lowest_price[i] = float(array_price[i, 1])
    personal_average_price[i] = float(array_price[i, 2])
plt.figure(1,figsize=(12,9))
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Price Trend")
plt.plot(t, personal_lowest_price, "r-", linewidth=0.8, label="personal_lowest_price")  # 个人最低成交价
plt.scatter(t,personal_lowest_price,s = 10,alpha = 1.0,marker ="o")
plt.plot(t, personal_average_price, "b-.", linewidth=0.6, label="personal_average_price")  # 个人平均成交价
plt.scatter(t,personal_average_price,s = 10,alpha = 0.8,marker="o")
plt.legend()
plt.show()

# Function to get data
def get_data(file_name):
    data = pd.read_csv(file_name)
    flash_x_parameter = []
    flash_y_parameter = []
    # arrow_x_parameter = []
    arrow_y_parameter = []
    for x1, y1, y2 in zip(data['date'], data['personal_lowest_price'], data['personal_average_price']):
         flash_x_parameter.append([float(x1)])
         flash_y_parameter.append(float(y1))
         arrow_y_parameter.append(float(y2))
         return flash_x_parameter, flash_y_parameter, arrow_y_parameter


# Function to know which Tv show will have more viewers
def more_viewers(x1, y1, y2):
    regr1 = linear_model.LinearRegression()   #regression:回归
    regr1.fit(x1, y1)
    predicted_value1 = regr1.predict(9)
    print(predicted_value1)
    regr2 = linear_model.LinearRegression()
    regr2.fit(x1, y2)
    predicted_value2 = regr2.predict(9)
    print(predicted_value2)
    # print predicted_value1
    # print predicted_value2
    # if predicted_value1 > predicted_value2:
    #     print("The Flash Tv Show will have more viewers for next week")
    # else:
    #     print("Arrow Tv Show will have more viewers for next week")

x1, y1,  y2 = get_data('data.csv')
# print x1,y1,x2,y2
more_viewers(x1, y1, y2)
