import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 
import pdb

df = pd.read_csv("HousePrices.csv")
print(df)
#pdb.set_trace()

plt.scatter(df.area,df.price,color='r', marker='+')
plt.xlabel('Area (m^2)',fontsize=20)
plt.ylabel('Prices ($)',fontsize=20)

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
print(reg.coef_)
print(reg.intercept_)


df2 = pd.read_csv("Area.csv")
pprices=reg.predict(df2)
df2['price']=pprices
print(df2)
df2.to_csv('PredictedPrices.csv',index=False)

plt.plot(df.area,reg.predict(df[['area']]),color='b')

plt.show()
