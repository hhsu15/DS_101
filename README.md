# Data Science 101

## Statistics Review

### Linear Algebra

Scalars: single number
Vectors: sequence of numbers - $\vec{u}$
matrix: a rectangular array of numbers with m rows and n columns.


Matrix Algebra:

- addtions: matrix_a + matrix_b -> matrix_c

Scalar Multiplication:

E.g., everything * 2

Dot Product:

The main idea is it outputs a single value.

E.g.,
```
np.dot(np.array([1, 3, 7]), np.array([2, 1, 3]))

= 1 * 2 + 3 * 1 + 7 * 3 = 26
```

Matrix Multiplication:

Matrix multiplication is only valid when you have something like:

(2 * 3 matrix) vs (3 * 2 matrix) 

The output will be a matrix.

Example:
```
[[1, 2, 3],      [[7, 8],
 [4, 5, 6]]   X   [9, 10],
                  [11, 12] 
                 ]
1 * 7 + 2 * 9 + 3 * 11 = 58  (up left)
1 * 8 + 2 * 10 + 3 * 12 = 64 (up right)
4 * 7 + 5 * 9 + 6 * 11 = 139 (down left)
4 * 8 + 5 * 10 + 6 * 12 = 154 (down right)
=>
[[58, 64],
 [139, 154]
]

```

Vector Norm

*vector norm* is the magnitude of a vector - showing the distance from the origin.

E.g., vactor norm of [3, 4] = sqrt(3^2 + 4^2) = 5


Measures of Central Tendency

- Mean, Median, Mode

Mean Squared Error (MSE)

- the distance between the predicton and actual
- same as how you calculate variance 
```
sum((actual - predict)^2) / len(y)
```

Root Mean Squared Error (RMSE)
- same as how you calculate standard deviation

## Model Bias and Variance

bias: shows how accurate a model is in its predictions. Low bias if the results are centered in the bull eye.

variance: shows ho reliable(consistent) a model is in its performance. Low variance if the points are close to each other

Mean Squared Error = bias + variance

- Directed Acyclic Graph

A directed acyclic graph (DAG) can help determine which variables are most important for your model. It helps to visually demonstrate the logic of your models.

A DAG always includes at least one exposure/predictor and one outcome.

To exam if there is a relationship between your features and target visually, you can use sns. pairplot.


- Statistical significance

Statistical significance is the likelihood that a result or relationship is caused by something other than mere random chance.


## Linear Regression

basic steps example, use temparature to predict rental
```python
from sklearn.linear_model import LinearRegression
feature_cols = ['temp']
X = bikes[feature_cols] 
y = bikes.total_rentals
lr = LinearRegression()
lr.fit(X, y) # train the model
lr.predict(np.array([0]).reshape(1, -1)) # apply new data (has to be two dimentional)
# predict two new data
lr.predict([[0], [10])

# print coefficients
print(lr.intercept_)
print(lr.coef_)
```

You can interpret the coefficients if you have:

'temp', 7.8648249924774394

This would mean increasing the tempeature by 1 unit will increate the rental by 7.86 bikes.

```

- Evaluation Metrics

-- Mean absolute error (MAE): mean of absolute value of the errors
-- Mean squared error (MSE): mean of squared errors
- Root mean squared error (RMSE): square root of the mean of squared errors.
```python
from sklearn import metrics
import numpy as np
print 'MAE:', metrics.mean_absolute_error(true, pred)
print 'MSE:', metrics.mean_squared_error(true, pred)
print 'RMSE:', np.sqrt(metrics.mean_squared_error(true, pred))

```

All of these are loss functions, *because we want to minimize them.*


## Feature Engineering to Improve Performance

Modify our features to have relationships that our models can understand. 
- convert categorical features (e.g., 1, 2, 3, 4)
- or use dummy ecoding (0/1) so it's unordered
```ptyhon

dummy_df = pd.get_dummies(categorical_feature, prefix="your_prefix")
```


- Regularization

To minimize overfitting, contraning the size of the coeffieients by shrinking them toward zero.

Two regularized linear regression model:
- Ridge regression
- Lasso regression 

- A larger alpha results in more regularization
```python
# alpha=0 is equivalent to linear regression.
from sklearn.linear_model import Ridge

# Instantiate the model.
#(Alpha of zero has no regularization strength, essentially a basic linear regression.)
ridgereg = Ridge(alpha=0.0, normalize=True)

```

## K-Nearest Neighbors

knn is a classification algo. It uses "k" most similr observations in order to make a prediction.

1. pick a value of k
2. search for k observations that are nearest to the measurements of the new input. - use Euclidian distance and maybe other metrics as well
3. Pick the most popular response from the neighbors and the predicted response

If k is 1, then basically just find the closest neighbor and use it as a response.

As the k increases, the prediction should be more distinct (more generalized).

Outliers affect less when k is larger.
When increasing k, the bias increses (less accurate) but variance decreases (less scattered)

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred_class = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred_class))

Compare testing accuracy with Null Accuracy

A technique to create a benchmark by alwasys predicting the most frequest class.
```python
y_test.value_counts()
y_test.value_counts().head(1) / len(y_test)  # assume the first one is the most frequest class


```


```python`
# Calculate predicted probabilities of class membership.
knn.predict_proba(X)

```

To measure the performance of different k, we can collect the data for training error (1 - accuracy) and testing error. If the training error is lower than testing error then the model is likely to be overfitting.

To find the optimal k, we can use GridSearchCV, e.g., 

```python
# This is very helpful!!!

from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()
param_grid = {'n_neighbors':[1,2,3,4,5,6,7,8,10],
#               'parameter2':[1,2,3,4]
             }
gs = GridSearchCV(estimator=knn,
                  param_grid = param_grid,
                  cv=5
                 )
gs.fit(X,y)

```

Standardize the scale is very important because for example KNN uses Euclidean distance metric so to determine coloseness the squared sum has to make sense, i.e., have the same scale across axises.

We can use StandardScaler to standardize data.
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

```

### Logistic Regression


