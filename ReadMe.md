# Stock Market Analysis

We will be looking at and analyzing data from the stock market. Using pandas, seaborn, and matplotlib to get the stock information and analysis, visualize different aspects of analysis, and analyze risk of various stocks using previous performance history.  Also use Monte Carlo method to predict future stock prices.


```python
#Install data reader to get stock info from yahoo
#pip install pandas-datareader
```


```python
#Imports
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import scipy.stats as stats

#for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# For reading stock data from yahoo
import pandas_datareader as pdr
from pandas_datareader import data as wb

# For time stamps
from datetime import datetime

# For division
from __future__ import division
```

## Questions to analyze

1) What was the change in price of the stock over time?

2) What was the daily return of the stock on average?

3) What was the moving average of the various stocks?

4) What was the correlation between different stocks closing prices?

5) What was the correlation between different stocks daily returns?

6) How much value do we put at risk by investing in a particular stock?

7) How can we attempt to predict future stock behavior?

### Basic Analysis of Stock Infromation

Use Yahoo and pandas to get data from some tech stocks


```python
#Create list of tech stocks to analyze
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
```


```python
#Setup start and end times for data grab
end = datetime.now()
start = datetime(end.year-1,end.month,end.day) 
```


```python
#Grab data
AAPL = pdr.get_data_yahoo('AAPL', start, end)
GOOG = pdr.get_data_yahoo('GOOG', start, end)
MSFT = pdr.get_data_yahoo('MSFT', start, end)
AMZN = pdr.get_data_yahoo('AMZN', start, end)
```


```python
#Summary stats for Apple
AAPL.describe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>252.000000</td>
      <td>252.000000</td>
      <td>252.000000</td>
      <td>252.000000</td>
      <td>2.520000e+02</td>
      <td>252.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>120.973234</td>
      <td>117.905446</td>
      <td>119.555298</td>
      <td>119.449712</td>
      <td>1.245264e+08</td>
      <td>119.054349</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.137854</td>
      <td>12.947596</td>
      <td>13.155370</td>
      <td>13.017755</td>
      <td>5.393529e+07</td>
      <td>13.135242</td>
    </tr>
    <tr>
      <th>min</th>
      <td>86.419998</td>
      <td>83.144997</td>
      <td>83.312500</td>
      <td>84.699997</td>
      <td>4.669130e+07</td>
      <td>84.133095</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>115.982500</td>
      <td>112.657499</td>
      <td>114.520002</td>
      <td>114.832500</td>
      <td>8.819405e+07</td>
      <td>114.269588</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>123.649998</td>
      <td>120.369999</td>
      <td>122.164997</td>
      <td>121.869999</td>
      <td>1.113192e+08</td>
      <td>121.572643</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>129.727501</td>
      <td>126.772501</td>
      <td>128.540001</td>
      <td>127.950003</td>
      <td>1.508143e+08</td>
      <td>127.717728</td>
    </tr>
    <tr>
      <th>max</th>
      <td>145.089996</td>
      <td>141.369995</td>
      <td>143.600006</td>
      <td>143.160004</td>
      <td>3.743368e+08</td>
      <td>142.704010</td>
    </tr>
  </tbody>
</table>

```python
#General info for Apple
AAPL.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 252 entries, 2020-06-12 to 2021-06-11
    Data columns (total 6 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   High       252 non-null    float64
     1   Low        252 non-null    float64
     2   Open       252 non-null    float64
     3   Close      252 non-null    float64
     4   Volume     252 non-null    float64
     5   Adj Close  252 non-null    float64
    dtypes: float64(6)
    memory usage: 13.8 KB


Weve looked at what included in the DataFrame, so lets plot out the volume and closing price of Apple


```python
#Visualization of past closing prices
AAPL['Adj Close'].plot(legend=True,figsize=(10,4))
```




![png](images/output_14_1.png)
 



```python
#Plot of total volume of stock being traded each day
AAPL['Volume'].plot(legend=True,figsize=(10,4))
```


![png](images/output_15_1.png)
    


Weve looked at the closing price and volumes traded each day for Apple over the last year. Now lets calculate the moving average for the close over 10, 20, and 50 days


```python
#Calculate moving averages for 10, 20, 50 days
ma_day = [10,20,50]

for ma in ma_day:
    column_name = 'MA for %s days' %(str(ma))
    
    AAPL[column_name] = AAPL['Adj Close'].rolling(ma).mean()
```

Now that we have moving averages calculated, lets visualize them.


```python
AAPL[['Adj Close','MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(subplots=False,figsize=(10,4))
```




![png](images/output_19_1.png)
    


### Daily Return Analysis

After looking at a basic analysis of Apple, lets begin to look at the risk of the stock.  We will need to begin by looking at the daily returns.


```python
#Use pct_change to calculate the percent change each day
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
```


```python
#Plot the daily return percentage
AAPL['Daily Return'].plot(figsize=(10,4), legend=True, linestyle='--',marker='o')
```


![png](images/output_23_1.png)
    


Now let use seaborn to get a histogram and kde plot for the same daily return percentages


```python
#Plot histogram and use dropna to remove all NaN values
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')
```


![png](images/output_25_2.png)
    

```python
#Another histogram plot of Apple
AAPL['Daily Return'].hist(bins=100)
```


![png](images/output_26_1.png)
    


Lets look at all the tech stocks in our list


```python
#Make a new dataframe of all the closing prices of the tech stocks in our list
closing_df = pdr.get_data_yahoo(tech_list, start, end)['Adj Close']
```


```python
#Preview dataframe
closing_df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>AAPL</th>
      <th>GOOG</th>
      <th>MSFT</th>
      <th>AMZN</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-06-12</th>
      <td>84.133095</td>
      <td>1413.180054</td>
      <td>185.939621</td>
      <td>2545.020020</td>
    </tr>
    <tr>
      <th>2020-06-15</th>
      <td>85.173584</td>
      <td>1419.849976</td>
      <td>187.128113</td>
      <td>2572.679932</td>
    </tr>
    <tr>
      <th>2020-06-16</th>
      <td>87.430885</td>
      <td>1442.719971</td>
      <td>191.713745</td>
      <td>2615.270020</td>
    </tr>
    <tr>
      <th>2020-06-17</th>
      <td>87.309204</td>
      <td>1451.119995</td>
      <td>192.377289</td>
      <td>2640.979980</td>
    </tr>
    <tr>
      <th>2020-06-18</th>
      <td>87.343979</td>
      <td>1435.959961</td>
      <td>194.437363</td>
      <td>2653.979980</td>
    </tr>
  </tbody>
</table>
We have the DataFrame of the closing prices, now lets get the daily returns of these stocks


```python
tech_rets = closing_df.pct_change()
```

Now we can compare the daily percentage return of two stocks.

First lets look at Google compared to itself


```python
sns.jointplot('GOOG','GOOG',tech_rets,kind = 'scatter',color='seagreen')
```


![png](images/output_34_2.png)
    


As expected, we see a perfect and positive correlation since we are comparing the same stock to each other.

Now lets compare Google and Microsoft to each other in the same way to view their relationship


```python
sns.jointplot('GOOG','MSFT',tech_rets,kind = 'scatter',color='seagreen')
```


![png](images/output_37_2.png)
    


We see a positive correlation between the two stocks. Lets check out this with all the tech stocks at once.


```python
#Visualize all stocks in list at once with a pair plot
sns.pairplot(tech_rets.dropna())
```


![png](images/output_39_1.png)
    


All relationships show a positive correlation. We will run a correlation test to get their pearson values to help determine most direct relationship but all stocks daily returns appear to be similar relationship to each other.

Lets use PairGrid to get a kdeplot included in the grid to investigate further


```python
#Create pairgrid to includes kdeplots
returns_fig = sns.PairGrid(tech_rets.dropna())

returns_fig.map_upper(plt.scatter,color='purple')

returns_fig.map_lower(sns.kdeplot,cmap='cool_d')

returns_fig.map_diag(sns.histplot,bins=30)
```


![png](images/output_42_1.png)
    


Nows lets do the same with the closing prices


```python
#Use pair grid again to visualize closing prices
returns_fig = sns.PairGrid(closing_df)

returns_fig.map_upper(plt.scatter,color='purple')

returns_fig.map_lower(sns.kdeplot,cmap='cool_d')

returns_fig.map_diag(sns.histplot,bins=30)
```


![png](images/output_44_1.png)
    


See an overall positive relationsihp between all stocks.  Amazons relationship were more neutral than other due to their closing price being fairly stable over last year.  See a very direct relationsihp between google and microsoft here

Lets get the correlation data between the stocks


```python
#Create correlation data for daily returns
corr_rets = tech_rets.corr()

corr_rets
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>AAPL</th>
      <th>GOOG</th>
      <th>MSFT</th>
      <th>AMZN</th>
    </tr>
    <tr>
      <th>Symbols</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AAPL</th>
      <td>1.000000</td>
      <td>0.505557</td>
      <td>0.688174</td>
      <td>0.685343</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>0.505557</td>
      <td>1.000000</td>
      <td>0.700824</td>
      <td>0.624505</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>0.688174</td>
      <td>0.700824</td>
      <td>1.000000</td>
      <td>0.738924</td>
    </tr>
    <tr>
      <th>AMZN</th>
      <td>0.685343</td>
      <td>0.624505</td>
      <td>0.738924</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>

```python
#Plot correlation data using a heatmap
sns.heatmap(corr_rets,annot=True)
```


![png](images/output_48_1.png)
    


As expected from the visualizations, all stocks had a high positive correlation between their daily returns with microsoft and amazon being the highest

Lets look at the correlations for the closing prices now


```python
#Create correlation data for closing prices
corr_close = closing_df.corr()

corr_close
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>AAPL</th>
      <th>GOOG</th>
      <th>MSFT</th>
      <th>AMZN</th>
    </tr>
    <tr>
      <th>Symbols</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AAPL</th>
      <td>1.000000</td>
      <td>0.672805</td>
      <td>0.743300</td>
      <td>0.772655</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>0.672805</td>
      <td>1.000000</td>
      <td>0.946116</td>
      <td>0.458864</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>0.743300</td>
      <td>0.946116</td>
      <td>1.000000</td>
      <td>0.620064</td>
    </tr>
    <tr>
      <th>AMZN</th>
      <td>0.772655</td>
      <td>0.458864</td>
      <td>0.620064</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>

```python
#Plot correlation data using a heatmap
sns.heatmap(corr_close,annot=True)
```


![png](images/output_52_1.png)
    


Again, we get what we expected. All postiive correlations with google and microsoft being very closely correlated at .95

Weve seen the positive relationships between these tech stocks over the last year, now lets look at a risk analysis

## Risk Analysis

Many ways to quantify risk.  One of the most basic ways is by using the daily percentage returns data already gather and using the expected return with the standard deviation of the daily returns


```python
#Get a new DataFrame and drop NaN
rets = tech_rets.dropna()

area = np.pi*20

plt.scatter(rets.mean(), rets.std(), alpha = 0.5,s=area)

#Set x and y limits
plt.ylim([0.01,0.025])
plt.xlim([-0.003,0.004])

#Set plot axis titles
plt.xlabel('Expected returns')
plt.ylabel('Risk')

#Label scatter plot
#http://matplotlib.org/users/annotations_guide.html
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (15, 10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))
```


![png](images/output_57_0.png)
    


See Google seems to have the highest expected return with less risk than apple or amazons stock using this method

Lets treat "value at risk" as the amount of money we could expect to lose (aka putting at risk) for a given C.I.

**Value at risk using the "bootstrap" method**
For this method we will calculate empirical quantiles from a histogram of daily returns. 



```python
#Plot the historgram of Apple again
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')
```


![png](images/output_60_2.png)
    


Now we can use the quantile to get the risk value for the stock


```python
#0.05 empirical quantile of daily returns
rets['AAPL'].quantile(0.05)
```


    -0.03315361479370249

The 0.05 empirical quantile of daily returns is -0.033.  This means that with 95% confidence, the worst daily loss will not exceed 3.3%.  Thus if we invested 1 million dollars into apple, the 1-day 5% VaR is 0.033*1,000,000 = $33,000.

Lets compare this with the other tech stocks


```python
quant = {}
for s in tech_list:
    q = rets[s].quantile(0.05)
    quant[s] = q
```


```python
quant
```


    {'AAPL': -0.03315361479370249,
     'GOOG': -0.02820798528720614,
     'MSFT': -0.027499588639915584,
     'AMZN': -0.02994135982097096}



We see similar risks between the stocks using the bootstrap method as well with apple being the highest risk and microsoft the lowest

**Risk using the monte carlo method**

Using the Monte Carlo to run many trials with random market conditions, then we'll calculate portfolio losses for each trial. After this, we'll use the aggregation of all these simulations to establish how risky the stock is.

Let's start with a brief explanation of what we're going to do:

We will use the geometric Brownian motion (GBM), which is technically known as a Markov process. This means that the stock price follows a random walk and is consistent with (at the very least) the weak form of the efficient market hypothesis (EMH): past price information is already incorporated and the next price movement is "conditionally independent" of past price movements.

This means that the past information on the price of a stock is independent of where the stock price will be in the future, basically meaning, you can't perfectly predict the future solely based on the previous price of a stock.

The equation for geometric Browninan motion is given by the following equation:

$$\frac{\Delta S}{S} = \mu\Delta t + \sigma \epsilon \sqrt{\Delta t}$$
Where S is the stock price, mu is the expected return (which we calculated earlier),sigma is the standard deviation of the returns, t is time, and epsilon is the random variable.

We can mulitply both sides by the stock price (S) to rearrange the formula and solve for the stock price.

$$ \Delta S = S(\mu\Delta t + \sigma \epsilon \sqrt{\Delta t}) $$
Now we see that the change in the stock price is the current stock price multiplied by two terms. The first term is known as "drift", which is the average daily return multiplied by the change of time. The second term is known as "shock", for each tiem period the stock will "drift" and then experience a "shock" which will randomly push the stock price up or down. By simulating this series of steps of drift and shock thousands of times, we can begin to do a simulation of where we might expect the stock price to be.

To demonstrate this, we will do just a few simualtions using the GOOG dataframe


```python
# Set up our time horizon
days = 365

# Now our delta
dt = 1/days

# Now let's grab our mu (drift) from the expected return data we got for AAPL
mu = rets.mean()['GOOG']

# Now let's grab the volatility of the stock from the std() of the average return
sigma = rets.std()['GOOG']
```

Create a function that will be able to input the starting price and number of days, as well as the already calculated sigma and mu


```python
def stock_monte_carlo(start_price, days, mu, sigma):
    
    #Define a price array
    price = np.zeros(days)
    price[0] = start_price
    
    #Shock and drift
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    #Run price array for number of days
    for x in range(1,days):
        #Calculate shock
        shock[x] = np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))
        #Calculate drift
        drift[x] = mu *dt
        #Calculate price
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
    return price
```


```python
GOOG.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-06-12</th>
      <td>1437.000000</td>
      <td>1386.020020</td>
      <td>1428.489990</td>
      <td>1413.180054</td>
      <td>1946400</td>
      <td>1413.180054</td>
    </tr>
    <tr>
      <th>2020-06-15</th>
      <td>1424.800049</td>
      <td>1387.920044</td>
      <td>1390.800049</td>
      <td>1419.849976</td>
      <td>1503900</td>
      <td>1419.849976</td>
    </tr>
    <tr>
      <th>2020-06-16</th>
      <td>1455.020020</td>
      <td>1425.900024</td>
      <td>1445.219971</td>
      <td>1442.719971</td>
      <td>1709200</td>
      <td>1442.719971</td>
    </tr>
    <tr>
      <th>2020-06-17</th>
      <td>1460.000000</td>
      <td>1431.380005</td>
      <td>1447.160034</td>
      <td>1451.119995</td>
      <td>1549600</td>
      <td>1451.119995</td>
    </tr>
    <tr>
      <th>2020-06-18</th>
      <td>1451.410034</td>
      <td>1427.010010</td>
      <td>1449.160034</td>
      <td>1435.959961</td>
      <td>1581900</td>
      <td>1435.959961</td>
    </tr>
  </tbody>
</table>

```python
#Get start price from previous cell
start_price=1428.48
for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
    
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for Google')
```


![png](images/output_75_1.png)
    


Lets get a histogram of the end results for a larger run.


```python
runs = 10000

simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]
```


```python
q = np.percentile(simulations,1)

plt.hist(simulations, bins=200)

# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# Title
plt.title(u"Final price distribution for Google Stock after %s days" % days, weight='bold');
```


![png](images/output_78_0.png)
    


Here we can see the 1% empirical quantile of the final price distribution to estiamte the Value at Risk for Google, which was %51.51 for every investment of 1428.48.  

So every share of Google invested in, there is about $51.51 at risk 99 percent of the time based on our monte carlo simulation.

Based on previous analysis, we should expect to see similar risk in the other previous tech stocks.

So lets look at a few other more recent volatile stocks.


## Risk management analysis of more volatile stocks

Lets use Tesla, Gamestop, and Target for other stocks to analyze risk


```python
#Create list of these stocks with their ticker
vol_list = ['TSLA','GME','TGT']
```


```python
#Grab data for each stock off yahoo
TSLA = pdr.get_data_yahoo('TSLA', start, end)
GME = pdr.get_data_yahoo('GME', start, end)
TGT = pdr.get_data_yahoo('TGT', start, end)
```

Now we need to get the closing price data and calculate the daily returns for each stock


```python
closing_df_vol = pdr.get_data_yahoo(vol_list, start, end)['Adj Close']
vol_rets = closing_df_vol.pct_change()
```


```python
vol_rets.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>TSLA</th>
      <th>GME</th>
      <th>TGT</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-06-12</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-06-15</th>
      <td>0.059469</td>
      <td>-0.006356</td>
      <td>0.010693</td>
    </tr>
    <tr>
      <th>2020-06-16</th>
      <td>-0.008851</td>
      <td>-0.010661</td>
      <td>0.008972</td>
    </tr>
    <tr>
      <th>2020-06-17</th>
      <td>0.009836</td>
      <td>0.025862</td>
      <td>-0.007885</td>
    </tr>
    <tr>
      <th>2020-06-18</th>
      <td>0.012271</td>
      <td>0.039916</td>
      <td>-0.008201</td>
    </tr>
  </tbody>
</table>

```python
#Get a new DataFrame and drop NaN
rets = vol_rets.dropna()

area = np.pi*20

plt.scatter(rets.mean(), rets.std(), alpha = 0.5,s=area)

#Set x and y limits
plt.ylim([0.01,.3])
plt.xlim([-0.001,.03])

#Set plot axis titles
plt.xlabel('Expected returns')
plt.ylabel('Risk')

#Label scatter plot
#http://matplotlib.org/users/annotations_guide.html
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (15, 10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))
```


![png](images/output_88_0.png)
    


As expected we see much higher returns and risk with GME with its recent run upwards of almost 5000% in a very short period.  


```python
quant_vol = {}
for s in vol_list:
    q = rets[s].quantile(0.05)
    quant_vol[s] = q
```


```python
quant_vol
```


    {'TSLA': -0.058358629618802116,
     'GME': -0.11077037970379816,
     'TGT': -0.01764185647450328}

Using the bootstrap method above we can again see higher risk in GME and then TSLA compared to TGT. Again with a 1 million dollar investment, with 95% confidence the largest amount of money lost in day in GME would be $110,770 compared to TGT at $17,000.

Lets look at the monte carlo simulation of TSLA next.


```python
TSLA.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-06-12</th>
      <td>197.595993</td>
      <td>182.520004</td>
      <td>196.000000</td>
      <td>187.056000</td>
      <td>83817000.0</td>
      <td>187.056000</td>
    </tr>
    <tr>
      <th>2020-06-15</th>
      <td>199.768005</td>
      <td>181.699997</td>
      <td>183.557999</td>
      <td>198.179993</td>
      <td>78486000.0</td>
      <td>198.179993</td>
    </tr>
    <tr>
      <th>2020-06-16</th>
      <td>202.576004</td>
      <td>192.477997</td>
      <td>202.369995</td>
      <td>196.425995</td>
      <td>70255500.0</td>
      <td>196.425995</td>
    </tr>
    <tr>
      <th>2020-06-17</th>
      <td>201.000000</td>
      <td>196.514008</td>
      <td>197.542007</td>
      <td>198.358002</td>
      <td>49454000.0</td>
      <td>198.358002</td>
    </tr>
    <tr>
      <th>2020-06-18</th>
      <td>203.839996</td>
      <td>198.893997</td>
      <td>200.600006</td>
      <td>200.792007</td>
      <td>48759500.0</td>
      <td>200.792007</td>
    </tr>
  </tbody>
</table>

```python
#Need to recalulate mu and sigma
mu = vol_rets.mean()['TSLA']

# Now let's grab the volatility of the stock from the std() of the average return
sigma = vol_rets.std()['TSLA']

#Get start price from previous cell
start_price=196
for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
    
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for Google')
```


![png](images/output_95_1.png)
    

```python
runs = 10000

simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]
```


```python
q = np.percentile(simulations,1)

plt.hist(simulations, bins=200)

# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# Title
plt.title(u"Final price distribution for Tesla Stock after %s days" % days, weight='bold');
```


![png](images/output_97_0.png)
    


Here we can see the 1% empirical quantile of the final price distribution to estiamte the Value at Risk for Tesla, which was $17.63 for every investment of 196.00 dollars.

So every share of Tesla invested in, there is about $17.63 at risk 99 percent of the time based on our monte carlo simulation.

With the 1% emprical quantile VaR of Google at abour 3 percent of the initial investment and Tesla at almost 9%, Google appears to be the less risky stock for an investment. 
