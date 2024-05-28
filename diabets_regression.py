from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

#Χρήση του Diabetes dataset:
diabetes = datasets.load_diabetes()
X = diabetes.data
Y = diabetes.target

#Διαχωρισμός της data σε train και test:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("X_test shape:", X_test.shape)
print("X_train shape:", X_train.shape)


#Εκπαίδευση του μοντέλου:
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

#Αξιολόγηση του μοντέλου:
print('Coefficients:', model.coef_) # η συναρτηση ως γραμμικος συνδυασμος των feature
print('Intercept:', model.intercept_) # o σταθερος ορος που εχει η συναρτηση
print('Mean squared error (MSE): %.2f' % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f' % r2_score(Y_test, Y_pred))

#Οπτικοποίηση των αποτελεσμάτων:
sns.scatterplot(x=Y_test, y=Y_pred)
plt.title('Scatter Plot of Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()
