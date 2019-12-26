import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 
import pdb
import math
from word2number import w2n

df = pd.read_csv("HousePrices.csv")
print(df)
median_bedrooms = math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(median_bedrooms)

df.bathrooms = df.bathrooms(pd.to_numeric)

#df.bathrooms = w2n.word_to_num(df['bathrooms'])
print(df)

#pdb.set_trace()

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
print(reg.coef_)
print(reg.intercept_)


pprice=reg.predict(df[['area','bedrooms','age']])
df['reg_price']=pprice
df['Residual']=df['price']-df['reg_price']
print(df)

#df2.to_csv('PredictedPrices.csv',index=False)

#plt.plot(df.area,reg.predict(df[['area']]),color='b')

#plt.show()
