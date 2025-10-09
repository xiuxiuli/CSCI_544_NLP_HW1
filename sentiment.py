import re
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import gensim.downloader as loader
import torch
import torch.nn as nn
import torch.optim as optim


# from sklearn.exceptions import UndefinedMetricWarning
# import warnings
# warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# import os

# nltk.download("stopwords", quiet=True)
# nltk.download("wordnet", quiet=True)
# nltk.download("omw-1.4", quiet=True)


# from tqdm import tqdm
# tqdm = lambda x, *args, **kwargs: x

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

PATH = "reviews.tsv"
SAMPLE_SIZE = 100000

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim=100):
        super(FeedForwardNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.net(x)

def data_prepare():
    # read file
    df = pd.read_csv(
        PATH,
        sep="\t",
        dtype=str,  
        quoting=csv.QUOTE_NONE,
        on_bad_lines="warn",
        engine="c",
    )

    # rename column
    df = df[["review_body", "star_rating"]].rename(
        columns = {"review_body":"review", "star_rating":"rating"}
    )

    # convert rating into number
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")  # errors="coerce" handles bad data to 'NaN'
    df = df.dropna(subset=["review", "rating"]) # drop those value='NaN'

    # before data wash, print qyt of Positive / Negative / Neutral
    pos_count = (df["rating"] > 3).sum()
    neg_count = (df["rating"] <= 2).sum()
    neu_count = (df["rating"] == 3).sum()

    # print as required in iii
    print(f"Positive reviews: {pos_count}")
    print(f"Negative reviews: {neg_count}")
    print(f"Neutral reviews: {neu_count}")

    # drop Neutral
    df = df[df["rating"] != 3]

    # gen label column (positive=1, negative=0)
    df["label"] = (df["rating"] > 3).astype(int)
    
    return df

def sampling(data, SAMPLE_SIZE=100000):
    # take pos and neg seperately
    pos_df = data[data["label"] == 1]
    neg_df = data[data["label"] == 0]

    # each sample 100,000 - balance
    pos_sample = pos_df.sample(n=min(SAMPLE_SIZE, len(pos_df)), random_state=42 )
    neg_sample = neg_df.sample(n=min(SAMPLE_SIZE, len(neg_df)), random_state=42
    )

    # merge: 100,000 pos + 100,000 neg
    balanced_df = pd.concat([pos_sample, neg_sample], axis=0).reset_index(drop=True)

    # 80/20 
    X_train, X_test, Y_train, Y_test = train_test_split(
        balanced_df["review"].tolist(),             # X feature
        balanced_df["label"].astype(int).values,    # Y label
        test_size=0.2,
        random_state=42,
        stratify=balanced_df["label"].values
    )

    return X_train, X_test, Y_train, Y_test

def save_split(X_train, X_test, Y_train, Y_test):
    # convert DataFrame
    train_df = pd.DataFrame({"review": X_train, "label": Y_train})
    test_df = pd.DataFrame({"review": X_test, "label": Y_test})
    
    # save locally
    train_df.to_csv("train_sample.csv", index=False)
    test_df.to_csv("test_sample.csv", index=False)
    
    print("✅ Saved train_sample.csv and test_sample.csv")

def data_clean(dataset, is_train=True):
    avg_before = float(np.mean([len(x) for x in dataset]))

    data_clean = [clean_text(x) for x in dataset]

    avg_after = float(np.mean([len(x) for x in data_clean]))

    if is_train:
        print(f"Average length before cleaning: {avg_before:.4f}")
        print(f"Average length after cleaning: {avg_after:.4f}")

    return data_clean

def clean_text(text:str) -> str:
    # lower case 
    text = text.lower()

    # remove  HTML tags 
    text = re.sub(r"<.*?>", " ", text)

    # remove url tags 
    text = re.sub(r"http\S+|wwww\.\S+", " ", text)

    # keep a-z and white space
    text = re.sub(r"[^a-z\s]", " ", text)

    # remove extra white space
    text = re.sub(r"\s+", " ", text).strip()

    return text

def preprocess(dataset, is_train=True):
    processed_data = [preprocess_text(x) for x in dataset]

    avg_after = float(np.mean([len(x) for x in processed_data]))

    if is_train:
        print(f"Average length after preprocessing: {avg_after:.4f}")

    return processed_data

def preprocess_text(text: str) -> str:
    # split
    tokens = text.split()

    # remove stop words
    tokens = [ w for w in tokens if w not in stop_words]

    # lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)

def feature_extraction(X_train_prep, X_test_prep):
    # new TF-IDT vectorizer
    vectorizer = TfidfVectorizer()

    # fit_trans on train set
    X_train_tfidf = vectorizer.fit_transform(X_train_prep)
    
    X_test_tfidf = vectorizer.transform(X_test_prep)

    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf, vectorizer

def run_perceptron(X_train_tfidf, y_train, X_test_tfidf, y_test):
    model = Perceptron(random_state=42)
    evaluate_model("Perceptron", model, X_train_tfidf, y_train, X_test_tfidf, y_test)

def run_svm(X_train_tfidf, y_train, X_test_tfidf, y_test):
    model = LinearSVC(random_state=42)
    evaluate_model("SVM", model, X_train_tfidf, y_train, X_test_tfidf, y_test)

def run_logistic_regression(X_train_tfidf, y_train, X_test_tfidf, y_test):
    model = LogisticRegression(max_iter=1000, random_state=42)
    evaluate_model("Logistic Regression", model, X_train_tfidf, y_train, X_test_tfidf, y_test)

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    # train
    model.fit(X_train, y_train)

    # predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    train_prec = precision_score(y_train, y_train_pred)
    train_rec = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"{name} Training Accuracy: {train_acc:.4f}")
    print(f"{name} Training Precision: {train_prec:.4f}")
    print(f"{name} Training Recall: {train_rec:.4f}")
    print(f"{name} Training F1-score: {train_f1:.4f}")
    print(f"{name} Testing Accuracy: {test_acc:.4f}")
    print(f"{name} Testing Precision: {test_prec:.4f}")
    print(f"{name} Testing Recall: {test_rec:.4f}")
    print(f"{name} Testing F1-score: {test_f1:.4f}")

def semantic_similarity_demo(model_glove):
    # (a) King - Man + Woman
    try:
        result1 = model_glove.most_similar(positive=["king", "woman"], negative=["man"], topn=4)
        words1 = ", ".join(["king"] + [w for w, _ in result1])
        print(f"king - man + woman = {words1}")
    except KeyError as e:
        print(f"Word not found in vocabulary: {e}")

    # (a) Top 5 similar words to "Outstanding"
    try:
        result2 = model_glove.most_similar("outstanding", topn=5)
        words2 = ", ".join([w for w, _ in result2])
        print(f"Outstanding = {words2}")
    except KeyError as e:
        print(f"Word not found in vocabulary: {e}")

def review_to_glove_avg(review, model):
    words = review.split()
    vectors = []

    for w in words:
        if w in model:
            vectors.append(model[w])
    
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    
    return np.mean(vectors, axis=0)

def review_to_glove_concat(review, model, max_words=10):
    words = review.split()[:max_words]
    vectors = []
    for w in words:
        if w in model:
            vectors.append(model[w])
        else:
            vectors.append(np.zeros(model.vector_size))
    
    while len(vectors) < max_words:
        vectors.append(np.zeros(model.vector_size))
    
    return np.concatenate(vectors, axis=0) 

def fast_glove_features(texts, model, feature_fn, input_dim=100):
    vocab = set(model.key_to_index)
    result = []
    for t in texts:
        words = [w for w in str(t).split() if w in vocab]
        if not words:
            # 动态返回匹配 input_dim 的全零向量
            result.append(np.zeros(input_dim))
        else:
            result.append(feature_fn(" ".join(words), model))
    return np.array(result)


def train_glove_fnn(model_glove, feature_fn, input_dim, tag, X_train, X_test, y_train, y_test):
    # ====== optimized, caceh features ======
    train_cache = f"{tag}_train.npy"
    test_cache  = f"{tag}_test.npy"

    try:
        X_train = np.load(train_cache)
        X_test  = np.load(test_cache)
    except FileNotFoundError:
        X_train = fast_glove_features(X_train, model_glove, feature_fn, input_dim)
        X_test  = fast_glove_features(X_test,  model_glove, feature_fn, input_dim)
        np.save(train_cache, X_train)
        np.save(test_cache, X_test)

    # if os.path.exists(train_cache) and os.path.exists(test_cache):
    #     X_train = np.load(train_cache)
    #     X_test  = np.load(test_cache)
    # else:
    #     X_train = fast_glove_features(X_train, model_glove, feature_fn, input_dim)
    #     X_test  = fast_glove_features(X_test, model_glove, feature_fn, input_dim)
    #     np.save(train_cache, X_train)
    #     np.save(test_cache, X_test)

    # 3. Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    y_test_t = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    # 4. Model definition
    model = FeedForwardNN(input_dim=input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5. Training loop
    if input_dim == 1000:
        EPOCHS = 2   # 拼接版本更慢
    else:
        EPOCHS = 5

    BATCH = 512
    n = len(X_train_t)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for i in range(0, n, BATCH):
            xb = X_train_t[i:i+BATCH]
            yb = y_train_t[i:i+BATCH]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # 6. Evaluation
    model.eval()
    with torch.no_grad():
        train_preds = torch.sigmoid(model(X_train_t)).numpy().round()
        test_preds = torch.sigmoid(model(X_test_t)).numpy().round()

    metrics(y_train, train_preds, f"{tag} Training")
    metrics(y_test, test_preds, f"{tag} Testing")

def metrics(y_true, y_pred, prefix):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{prefix} Accuracy: {acc:.4f}")
    print(f"{prefix} Precision: {prec:.4f}")
    print(f"{prefix} Recall: {rec:.4f}")
    print(f"{prefix} F1-score: {f1:.4f}")

def q3_logistic_regression():
    data = data_prepare()
    X_train, X_test, Y_train, Y_test = sampling(data, 100000)
    # save_split(X_train, X_test, Y_train, Y_test)

    # clean data
    X_train_clean = data_clean(X_train, is_train=True)
    X_test_clean = data_clean(X_test, is_train=False)

    # preprocess
    X_train_prep = preprocess(X_train_clean, is_train=True)
    X_test_prep = preprocess(X_test_clean, is_train=False)

    # TF-IDF
    X_train_tfidf, X_test_tfidf, vectorizer = feature_extraction(X_train_prep, X_test_prep)

    # train and evaluate models
    run_perceptron(X_train_tfidf, Y_train, X_test_tfidf, Y_test)

    run_svm(X_train_tfidf, Y_train, X_test_tfidf, Y_test)

    run_logistic_regression(X_train_tfidf, Y_train, X_test_tfidf, Y_test)

    return X_train_prep, X_test_prep, Y_train, Y_test

def q4_word_embed_fnn(X_train_prep, X_test_prep, Y_train, Y_test):
    model_glove = loader.load("glove-wiki-gigaword-100")

    # (a)
    semantic_similarity_demo(model_glove)

    # (b)
    train_glove_fnn(model_glove, feature_fn=review_to_glove_avg, input_dim=100, tag="Average Feature",
                    X_train=X_train_prep, X_test=X_test_prep,y_train=Y_train, y_test=Y_test)

    # (c)
    train_glove_fnn(model_glove, feature_fn=review_to_glove_concat, input_dim=1000, tag="Concatenated Feature",
                    X_train=X_train_prep, X_test=X_test_prep,y_train=Y_train, y_test=Y_test)

def main():
    X_train_prep, X_test_prep, Y_train, Y_test = q3_logistic_regression()
    q4_word_embed_fnn(X_train_prep, X_test_prep, Y_train, Y_test)

if __name__ == '__main__':
    main()
