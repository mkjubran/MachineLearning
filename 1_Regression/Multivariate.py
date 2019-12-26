import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 
import pdb
import math
from word2number import w2n

df = pd.read_csv("HousesPrices_Exercise.csv")
print(df)
median_bedrooms = math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(median_bedrooms)
print(df)

df.bathrooms = df.bathrooms.fillna('zero')
df.bathrooms = df.bathrooms.apply(w2n.word_to_num)
#median_bathrooms = math.floor(df.bathrooms.median())
#df.bathrooms = df.bathrooms.fillzero(median_bathrooms)
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age','bathrooms']],df.price)
print(reg.coef_)
print(reg.intercept_)


ppr=reg.predict(df[['area','bedrooms','age','bathrooms']])
df['reg_price']=ppr
df['Residual']=df['price']-df['reg_price']
print(df)

#df2.to_csv('PredictedPrices.csv',index=False)

#plt.plot(df.area,reg.predict(df[['area']]),color='b')

#plt.show()
