from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1) Load
imdb = load_dataset("imdb")
X_train, y_train = imdb["train"]["text"], imdb["train"]["label"]
X_test,  y_test  = imdb["test"]["text"],  imdb["test"]["label"]

pipe = make_pipeline(
    TfidfVectorizer(
        ngram_range=(1,2),
        max_features=50000,
        min_df=3,
        max_df=.95,
        stop_words='english'
    ),
    LogisticRegression(max_iter=1000, n_jobs=-1)
)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["neg","pos"]))