from src.utils import *

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet('data/raw_data/df.parquet.gzip')

print('Download train and test!')
x_train, x_test, y_train, y_test = download_train_test(df)

print('Vectorizing code!')
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train['code'])
x_test = vectorizer.transform(x_test['code'])

# print('Using SVD!')
# svd = TruncatedSVD(n_components=100)
# x_train = svd.fit_transform(x_train)
# x_test = svd.transform(x_test)

print('Training model!')
logreg = LogisticRegression(max_iter=1000, multi_class='ovr', verbose=1, n_jobs=10)
logreg.fit(x_train, y_train)
predicted = logreg.predict(x_test)
score = logreg.score(x_test, y_test)
logreg_score_ = np.mean(score)

print('Accuracy : %.3f' % (logreg_score_))

save_model(vectorizer, logreg)
