import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels import regression
from datetime import datetime
from iexfinance.stocks import get_historical_data
from iexfinance.stocks import Stock

# Retrive data from IEX Cloud Sandbox
start = datetime(2020, 5, 20)
end = datetime(2020, 10, 20)

asset1 = get_historical_data("ALK", start, end,  output_format='pandas', close_only=True)["close"]
asset1.name = 'ALK'
asset2 = get_historical_data("FISV", start, end,  output_format='pandas', close_only=True)["close"]
asset2.name = 'FISV'
benchmark = get_historical_data("SPY", start, end,  output_format='pandas', close_only=True)["close"]
benchmark.name = 'SPY'
print("***********" + str(Stock("ALK").get_company_name()) + "***********")
print(asset1)
print("***********" + str(Stock("FISV").get_company_name()) + "***********")
print(asset2)
print("***********" + str(Stock("SPY").get_company_name()) + "***********")
print(benchmark)

# First, run a linear regression on the two assets
slr = regression.linear_model.OLS(asset1, sm.add_constant(asset2)).fit()
print('SLR beta of asset2: ' + str(slr.params[1]))

# Run multiple linear regression using asset2 and SPY as independent variables
mlr = regression.linear_model.OLS(asset1, sm.add_constant(np.column_stack((asset2, benchmark)))).fit()

prediction = mlr.params[0] + mlr.params[1]*asset2 + mlr.params[2]*benchmark
prediction.name = 'Prediction'

print('MLR beta of asset2:  ' + str(mlr.params[1])) 
print('MLR beta of S&P 500: ' +  str(mlr.params[2]))
print(str(mlr.summary()))

# Plot the three variables along with the prediction given by the MLR
asset1.plot()
asset2.plot()
benchmark.plot()
prediction.plot(color='y')
plt.xlabel('Price')
plt.legend(bbox_to_anchor=(1,1), loc=2)

# Plot only the dependent variable and the prediction to get a closer look
asset1.plot()
prediction.plot(color='y')
plt.xlabel('Price')
plt.legend()
plt.show()