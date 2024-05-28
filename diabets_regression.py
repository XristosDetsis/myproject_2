from sklearn import datasets

diabetes = datasets.load_diabetes()

print(diabetes.DESCR)

X = diabetes.data
Y = diabetes.target

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print(X_test.shape)
print(X_train.shape)

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

model = linear_model.LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print(model)

print('Coefficients:', model.coef_) # η συναρτηση ως γραμμικος συνδυασμος των feature
print('Intercept:', model.intercept_) # o σταθερος ορος που εχει η συναρτηση
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))

import seaborn as sns


import seaborn as sns
import matplotlib.pyplot as plt

# Δημιουργία scatter plot
sns.scatterplot(x=Y_test, y=Y_pred)

# Προσθήκη τίτλου και ετικετών αξόνων
plt.title('Scatter Plot of Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Εμφάνιση του γραφήματος
plt.show()
