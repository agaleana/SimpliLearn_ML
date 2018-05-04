# -*- coding: utf-8 -*-
"""
CellStrat
"""

#==============================================================================
# First step to write the python program is to take benefit out of libraries
# already available. We will only focus on the data science related libraries.
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#==============================================================================
# #import data from the data file. In our case its Insurance.csv. 
#==============================================================================

insuranceData = pd.read_csv ('Insurance.csv')


#==============================================================================
# All mathematical operations will be performed on the matrix, so now we create
# matrix for dependent variables and independent variables.
#==============================================================================

X = insuranceData.iloc [:,0:1].values
y = insuranceData.iloc [:, 1].values

#==============================================================================
# Fit our data on Decision Tree Regressor
#==============================================================================
from sklearn.tree import DecisionTreeRegressor
DTregressor = DecisionTreeRegressor (random_state = 0)
DTregressor.fit (X, y)

#==============================================================================
# Visualize the regressor algo outcome
#==============================================================================
# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, DTregressor.predict(X), color = 'blue')
plt.title('Insurance Premium - Decision Tree')
plt.xlabel('Age')
plt.ylabel('Premium')
plt.show()

#==============================================================================
# Now see how accurately decision tree regressor predict premium based
# on age. Here values will be only exactly from the y array for certain range
# of values as we are taking average.
#==============================================================================
val = 68
predictionDT = DTregressor.predict (val)
print(predictionDT)
