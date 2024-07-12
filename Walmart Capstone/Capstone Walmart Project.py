#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r"G:\Intelipaat Rajani\Walmart DataSet.csv")
data.head(7)


# In[2]:


data.info()


# In[3]:


data.shape


# In[4]:


data.columns


# In[5]:


data.describe


# In[6]:


data.describe()


# In[7]:


plt.scatter(data.Fuel_Price,data.Weekly_Sales)
plt.show


# In[8]:


data.isna().sum()


# In[9]:


data.duplicated().sum()


# In[10]:


data.nunique()


# In[11]:


data['Weekly_Sales'].nunique()


# Therefore there are no null values and duplicates in the dataset.
# 
# a. If the weekly sales are affected by the unemployment rate, if yes - which stores
# are suffering the most?
# 

# In[12]:


data.set_index('Date',inplace = True)
data


# In[13]:


print(data['Weekly_Sales'].min())
print(data['Weekly_Sales'].max())


# In[14]:


sub_data = data.loc[:, ['Store', 'Weekly_Sales', 'Unemployment']];sub_data


# '''d1=data.groupby(['Store']).sum().reset_index()
# plt.figure(figsize=(15,5))
# sns.barplot(x='Store',y='Weekly_Sales', data=d1, order=d1.sort_values(by='Weekly_Sales',ascending=False)['Store'])'''

# In[15]:


correlations = sub_data.groupby('Store').corr()


# In[16]:


correlations = correlations.reset_index()
correlations


# In[17]:


correlations.drop(['level_1', 'Weekly_Sales'], axis = 1, inplace=True)
correlations.head()


# In[18]:


correlations.set_index(['Store'], inplace=True)
correlations.head()


# In[19]:


correlations.rename({'Unemployment':'Weekly_Sales vs Unemployment correlation'}, inplace = True, axis = 1)
correlations.head(8)


# In[20]:


correlations[(correlations['Weekly_Sales vs Unemployment correlation'] <= -0.4)]


# Stores 38 and 44 are affected most by unemployment.

# b. If the weekly sales show a seasonal trend, when and what could be the reason?

# In[21]:


import seaborn as sns
figure = plt.figure(figsize=(10,4))
plot = sns.lineplot(data=data, x = data.index, y = 'Weekly_Sales',label='weekly sales')
plt.show()


# c. Does temperature affect the weekly sales in any manner?

# In[22]:


plt.scatter(data['Weekly_Sales'], data['Temperature'])
plt.xlabel('Weekly Sales')
plt.ylabel('Temperature')
plt.title('Scatter Plot of Weekly Sales vs. Temperature')
plt.show()


# In[23]:


plt.figure(figsize=(15,5))
sns.barplot(data=data, x = "Temperature", y = 'Weekly_Sales',label='weekly sales')
plt.show()


# Temperature does not afffect the sales of the store.

# d. How is the Consumer Price index affecting the weekly sales of various stores?

# In[24]:


figure = plt.figure(figsize=(12,7))
sns.scatterplot(data = data, x = 'CPI', y = 'Weekly_Sales',hue='Store')
plt.show()


# CPI also does not affects the sales.

# e. Top performing stores according to the historical data.

# In[25]:


performance = data.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
top = performance.nlargest(5)
top


# store 20 is the top performer.

# f. The worst performing store, and how significant is the difference between the
# highest and lowest performing stores.

# In[26]:


worst = performance.nsmallest(3)
worst


# The worst performing store is 33.

# In[27]:


top-worst


# In[28]:


performance.iloc[0]-performance.iloc[-1]


# The difference in sales is around 18L . It is a big difference.

# In[29]:


from statsmodels.tsa.stattools import adfuller
result = adfuller(data['Weekly_Sales'])
print(f'p-value: {result[1]}')


# In[31]:


date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='W-FRI')


# In[32]:


'''
data['Weekly_Sales'] = pd.to_datetime(data['Weekly_Sales'])  #converting month column str to datetime
data.index = data['Weekly_Sales'] #making month column as a index column
del data['Weekly_Sales']
'''


# In[33]:


'''

#seasonal Decompose
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_result = seasonal_decompose(data['Weekly_Sales'].dropna(),freq='W-Fri')
decompose_result.plot();
#can't understand what is the error..tried different things too many

'''


# In[34]:


def P(pr):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(pr)
    print(f'p-value: {result[1]}')


# In[35]:


def acf_plot(fc):
    '''Plots an Autocorrelation Function for the given series'''
    from statsmodels.tsa.stattools import acf 
    from statsmodels.graphics.tsaplots import plot_acf

    acf_vals = acf(fc)
    plot_acf(acf_vals)
    plt.show()


# In[36]:


def pacf_plot(srs,nlags):
    '''Plots an Partial Autocorrelation Function for the given series'''
    from statsmodels.tsa.stattools import pacf
    from statsmodels.graphics.tsaplots import plot_pacf

    pacf_vals = pacf(srs)
    while True:
        try:
            plot_pacf(pacf_vals, lags = nlags)
            break
        except:
            plt.close('all')
            nlags -= 1


# Store 1 model

# In[37]:


s1 = data[data['Store'] == 1].loc[:,'Weekly_Sales']
s1 = pd.DataFrame(s1)
s1


# In[38]:


s1.index.freq = 'W-FRI'
s1.plot()


# In[30]:


#seasonal Decompose
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_result = seasonal_decompose(data['s1'].dropna(),period=7)
decompose_result.plot();


# In[39]:


s1['logx'] = np.log(s1['Weekly_Sales'])
s1['ma'] = s1['logx'].rolling(window=12).mean()
s1.dropna(inplace=True)
s1['dif'] = s1['logx'] - s1['ma']


# In[40]:


s1['dif2'] = s1['dif'].diff()
s1.dropna(inplace=True)


# In[41]:


P(s1['dif2'])


# In[42]:


acf_plot(s1['dif2'])


# In[43]:


pacf_plot(s1['dif2'], 10)


# In[44]:


train = s1['dif2'].iloc[:120]
test = s1['dif2'].iloc[120:]


# In[45]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
model1 = SARIMAX(train, order = (0,1,0), seasonal_order=(0,1,0,12))
fm1 = model1.fit()


# In[46]:


s1preds = fm1.predict(start = len(train),
                   end = len(train) + len(test) - 1,
                   dynamic = True)


# In[47]:


s1['dif2'].plot()
s1preds.plot()


# In[48]:


s1fcast = fm1.forecast(steps = 36)
s1['dif2'].plot()
s1fcast.plot()


# In[90]:


s1fcast


# Store 2 MODEL

# In[49]:


s2 = data[data['Store'] == 2].loc[:,'Weekly_Sales']
s2 = pd.DataFrame(s2)
s2


# In[50]:


s2.index.freq = 'W-FRI'
s2.plot()


# In[51]:


s2['logx'] = np.log(s2['Weekly_Sales'])
s2['ma'] = s2['logx'].rolling(window=12).mean()
s2.dropna(inplace=True)


# In[52]:


s2['dif'] = s2['logx'] - s2['ma']
s2['dif2'] = s2['dif'].diff()
s2.dropna(inplace=True)


# In[53]:


P(s2['dif2'])


# In[54]:


acf_plot(s2['dif2'])


# In[55]:


pacf_plot(s2['dif2'], 10)


# In[56]:


train = s2['dif2'].iloc[:120]
test = s2['dif2'].iloc[120:]


# In[57]:


model2 = SARIMAX(train, order = (1,2,1), seasonal_order=(1,2,1,12))
fm2 = model2.fit()


# In[58]:


s2preds = fm2.predict(start = len(train),
                   end = len(train) + len(test) - 1,
                   dynamic = True)


# In[59]:


s2['dif2'].plot()
s2preds.plot()


# In[60]:


s2fcast = fm2.forecast(steps = 36)


# In[61]:


s2['dif2'].plot()
s2fcast.plot()


# In[91]:


s2fcast


#  Store 19 MODEL

# In[62]:


s19 = data[data['Store'] == 19].loc[:,'Weekly_Sales']
s19 = pd.DataFrame(s19)
s19


# In[63]:


s19.index.freq = 'W-FRI'
s19.plot()


# In[64]:


s19['logx'] = np.log(s19['Weekly_Sales'])
s19['ma'] = s19['logx'].rolling(window=12).mean()
s19.dropna(inplace=True)
s19['dif'] = s19['logx'] - s19['ma']


# In[65]:


s19['dif2'] = s19['dif'].diff()
s19.dropna(inplace=True)
P(s19['dif2'])


# In[66]:


acf_plot(s19['dif2'])


# In[67]:


pacf_plot(s19['dif2'], 10)


# In[68]:


train = s19['dif2'].iloc[:120]
test = s19['dif2'].iloc[120:]


# In[69]:


model19 = SARIMAX(train, order = (2,1,0), seasonal_order=(2,1,0,12))
fm19 = model19.fit()


# In[70]:


s19preds = fm19.predict(start = len(train),
                   end = len(train) + len(test) - 1,
                   dynamic = True)


# In[71]:


s19['dif2'].plot()
s19preds.plot()


# In[72]:


s19fcast = fm19.forecast(steps = 36)
s19['dif2'].plot()
s19fcast.plot()


# In[92]:


s19fcast


# Store 20 MODEL

# In[73]:


s20 = data[data['Store'] == 20].loc[:,'Weekly_Sales']
s20 = pd.DataFrame(s20)
s20


# In[74]:


s20.plot()


# In[75]:


s20['logx'] = np.log(s20['Weekly_Sales'])
s20['ma'] = s20['logx'].rolling(window=12).mean()
s20.dropna(inplace=True)
s20['dif'] = s20['logx'] - s20['ma']
s20['dif2'] = s20['dif'].diff()
s20.dropna(inplace=True)
P(s20['dif2'])


# In[82]:


acf_plot(s20['dif2'])
pacf_plot(s20['dif2'],10)


# In[84]:


train = s20['dif2'].iloc[:120]
test = s20['dif2'].iloc[120:]
model20 = SARIMAX(train, order = (2,1,0), seasonal_order=(2,1,0,12))
fm20 = model20.fit()


# In[85]:


s20preds = fm20.predict(start = len(train),
                   end = len(train) + len(test) - 1,
                   dynamic = True)


# In[107]:


s20['dif2'].plot()
print(s20preds.plot())


# In[89]:


s20fcast = fm20.forecast(steps = 48)
s20['dif2'].plot()
s20fcast.plot()


# In[93]:


s20fcast

Store 4 MODEL
# In[97]:


s4=data[data['Store']==4].loc[:,'Weekly_Sales']
s4=pd.DataFrame(s4)
s4


# In[98]:


s4.plot()


# In[100]:


s4['logx'] = np.log(s4['Weekly_Sales'])
s4['ma'] = s4['logx'].rolling(window=12).mean()
s4.dropna(inplace=True)
s4['dif'] = s4['logx'] - s4['ma']
s4['dif2'] = s4['dif'].diff()
s4.dropna(inplace=True)
P(s4['dif2'])


# In[104]:


acf_plot(s4['dif2'])
pacf_plot(s4['dif2'],10)
train = s4['dif2'].iloc[:120]
test = s4['dif2'].iloc[120:]
model4 = SARIMAX(train, order = (2,1,0), seasonal_order=(2,1,0,12))
fm4 = model4.fit()

s4preds = fm4.predict(start = len(train),
                   end = len(train) + len(test) - 1,
                   dynamic = True)


# In[105]:


s4['dif2'].plot()


# In[106]:


s4preds.plot()


# In[109]:


s4fcast = fm4.forecast(steps = 36)
s4['dif2'].plot()
s4fcast.plot()


# In[110]:





# In[114]:


print(s4fcast)
s4fcast.plot()

