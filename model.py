import pandas as pd
data = pd.read_csv('spam.csv', encoding = 'latin-1')

data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)

data['label'] = data['v1'].map({'ham' : 0, 'spam' : 1})

X = data['v2']
y = data['label']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

import pickle
pickle.dump(cv, open('transform.pkl', 'wb'))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x_train, y_train)
clf.score(x_test, y_test)

filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))