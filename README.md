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

Logistic regression uses elements from both Linear regression and knn algo.

Since the values output from a linear regression cannot be interpreted as probabilities of class membership since the value can be greater than 1 and less than 0. Logistic regression on the other hand ensures that the values output as predictions can be interpreted as probabilities of class membership.

```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

feature_cols = ['al']
X = glass[feature_cols]
y = glass.household

logreg.fit(X,y)
pred = logreg.predict(X)

```

In addtion to the class prediction, logistic regression is very helpful in terms of predicting probabilities.

```python
logreg.predict_proba(X) # return a list  of probabilities for 0 and 1, e.g., [ 0.97193375,  0.02806625]

```

#### Probability, Odds Ratio, e, Log, and Log Odds

prob = one outcome/all outcomes
odds = one outcome / all other outcomes

- Dice roll of 1: probability = 1/6, odds = 1/5
- Even dice roll: probability = 3/6, odds = 3/3 = 1
- Dice roll less than 5: probability = 4/6, odds = 4/2 = 2

```python
# Create a table of probability versus odds.
table = pd.DataFrame({'probability':[0.0, 0.1, 0.2, 0.25, 0.5, 0.6, 0.8, 0.9,1.0]})
table['odds'] = table.probability/(1 - table.probability)
table

```

#### e and natural log

What is e? It is the base rate of growth shared by all continually growing processes:

- 2.718281828459

e is what is Eulers number and is irrational and is the base of the natural log, `ln`.b

What is a (natural) log? It gives you the time needed to reach a certain level of growth.

#### The Log Odds Ratio

You can the logarithm of the odds ratio and you get what's known as log odds. 

Odds ratio can never be negative.

Log odds has the range from negative inf to positive inf.


```python
# Add log odds to the table.
import numpy as np
table['logodds'] = np.log(table['odds'])
table

```

Logistic regress uses log odds as a categorical response being true is modeled as a linear combination of the features. 

$$\log \left({p\over 1-p}\right) = \beta_0 + \beta_1x$$

- So Logistic regression outputs the probabilities of a specific class.
- Those probabilities can be converted into class predictions.

## NLP

Task example: text classification (positive feedback or nagative), text extraction (understand topics)

In data science, we are often asked to analyze unstructured text or make a predictive model using it. Unfortunately, most data science techniques require numeric data. NLP libraries provide a tool set of methods to convert unstructured text into meaningful numeric data.

#### Lower level components

- Tokenization: word token, sentences, n-grams
- Stop-words removal
- Stemming and lemmatization: root word
- TD-IDF: word importance
- Part of speech tagging: noun/verb/adjective
- Named Entity Recognition (NER): person/organization/location
- Word sense disambiguation: buy a mouse
- Segmentation: "New York city subway"
- Languge detection: "translate this page"
- Machine learning: specialized models that work well with text

Naive Bayes is a popular algo for text classification.

Set up the dependencies,
```python
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB         # Naive Bayes
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer

%matplotlib inline

```
We may want to identify:

Is an article a sports or business story?
Does an email have positive or negative sentiment?
Is the rating of a recipe 1, 2, 3, 4, or 5 stars?


#### CountVectorizer

We will use CountVectorizer to convert each document into a set of words and their counts, just the the collections.Counter. 
So you don't have to do it mannually. You can use vect.vocabulary_to see the actual word counts.

```python
# Use CountVectorizer to create document-term matrices from X_train and X_test.
# Use default options for CountVectorizer.
vect = CountVectorizer()

# Create document-term matrices.
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# Use Naive Bayes to predict the star rating.
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)

# Calculate accuracy.
print(metrics.accuracy_score(y_test, y_pred_class))
```

Get the base line accuracy (assuming model always predicts the top category)
```python
# calculate null accuracy
y_test_binary = np.where(y_test==5, 1, 0) # five stars become 1, one stars become 0
print 'Percent 5 Stars:', y_test_binary.mean()
print 'Percent 1 Stars:', 1 - y_test_binary.mean()

```

#### Common preprocessing techniques

- N-Grams 

This can lead to more feature columns.
```python

# Include 1-grams and 2-grams.
vect = CountVectorizer(ngram_range=(1, 2))
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape
```

- remove stop words

```python
# Remove English stop words.
vect = CountVectorizer(stop_words='english')
vect.get_params()
vect.get_stop_words()
```

You can also specify max features
```python
# order by term frequency
# Remove English stop words and only keep 100 features.
vect = CountVectorizer(stop_words='english', max_features=100)
tokenize_test(vect)

```

#### You can use TextBlob to analyze and process the text

```python
review = TextBlob(yelp_best_worst.text[0])

# List the words.
review.words

# List the sentences.
review.sentences

# Some string methods are available.
review.lower()

stemmer = SnowballStemmer('english')

# Stem each word.
print([stemmer.stem(word) for word in review.words])

# Assume every word is a noun. This is in textbolb
print([word.lemmatize() for word in review.words])

# Assume every word is a verb.
print([word.lemmatize(pos='v') for word in review.words])


#### Term Frequency-Inverse Document Frequency (TF-IDF)

While a Count Vectorizer simply totals up the number of times a "word" appears in a document, the more complex TF-IDF Vectorizer analyzes the uniqueness of words between documents to find distinguishing characteristics.

Term frequency–inverse document frequency (TF–IDF) computes the "relative frequency" with which a word appears in a document, compared to its frequency across all documents.

It's more useful than "term frequency" for identifying "important" words in each document (high frequency in that document, low frequency in other documents).

It's used for search-engine scoring, text summarization, and document clustering.

```python
# Example documents
simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']

# Term frequency 
vect = CountVectorizer()
tf = pd.DataFrame(vect.fit_transform(simple_train).toarray(), columns=vect.get_feature_names())
tf

# Document frequency (each word appears in how many documents)
vect = CountVectorizer(binary=True)
df = vect.fit_transform(simple_train).toarray().sum(axis=0)
pd.DataFrame(df.reshape(1, 6), columns=vect.get_feature_names())
```

```python
# Term frequency–inverse document frequency (simple version)
tf/df

```

The higher the TF–IDF value, the more "important" the word is to that specific document. Here, "cab" is the most important and unique word in document 1, while "please" is the most important and unique word in document 2. TF–IDF is often used for training as a replacement for word count.

```python
# TfidfVectorizer
vect = TfidfVectorizer()
pd.DataFrame(vect.fit_transform(simple_train).toarray(), columns=vect.get_feature_names())

```

Often time, the TfidfVectorizer is better than simple CountVectorizer.

Example analysis:
```python
# Create a document-term matrix using TF–IDF.
vect = TfidfVectorizer(stop_words='english')

# Fit transform Yelp data.
dtm = vect.fit_transform(yelp.text)
features = vect.get_feature_names()

def summarize():
    
    # Choose a random review that is at least 300 characters.
    review_length = 0
    while review_length < 300:
        review_id = np.random.randint(0, len(yelp))
        review_text = yelp.text[review_id]
        #review_text = unicode(yelp.text[review_id], 'utf-8')
        review_length = len(review_text)
    
    # Create a dictionary of words and their TF–IDF scores.
    word_scores = {}
    for word in TextBlob(review_text).words:
        word = word.lower()
        if word in features:
            word_scores[word] = dtm[review_id, features.index(word)]
    
    # Print words with the top five TF–IDF scores.
    print('TOP SCORING WORDS:')
    top_scores = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for word, score in top_scores:
        print(word)
    
    # Print five random words.
    print('\n' + 'RANDOM WORDS:')
    random_words = np.random.choice(list(word_scores.keys()), size=5, replace=False)
    for word in random_words:
        print(word)
    
    # Print the review.
    print('\n' + review_text)
```

#### Sentiment Analysis

```python
# Define a function that accepts text and returns the polarity.
def detect_sentiment(text):
    return TextBlob(text.decode('utf-8')).sentiment.polarity
    #return TextBlob(text).sentiment.polarity

```

