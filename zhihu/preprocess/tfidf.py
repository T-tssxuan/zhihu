from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

tfidf = TfidfVectorizer(input='filename')
X = tfidf.fit_transform(['../data/question_train_word_set.csv'])
joblib.dump(X, '../data/filename.pkl') 
