import re
import time
from collections import defaultdict

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, classification_report


class Tokenizer:
    def __init__(self, vocabulary_size=1000):
        self.vocab_size = vocabulary_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_frequencies = {}  # words and their frequencies
        self.stop_words = set()  # words not added to vocabulary

    def load_stop_words(self, path):
        self.stop_words = set()
        with open(path, "r") as f:
            for line in f:
                self.stop_words.add(line.strip())

    def init_vocabulary(self, documents):
        vocabulary = {}
        for doc in documents:
            words = re.findall(r"\b[a-zA-Z]+\b", doc.lower())  # get all words
            for word in words:
                if word in self.stop_words:  # filter stop words
                    continue
                if word not in vocabulary:  # add word to vocabulary
                    vocabulary[word] = 1
                else:
                    vocabulary[word] += 1
        # sort vocabulary by frequency
        sorted_vocabulary = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_vocabulary) > self.vocab_size:
            sorted_vocabulary = sorted_vocabulary[: self.vocab_size]
        else:
            self.vocab_size = len(sorted_vocabulary)

        # init word_to_idx and idx_to_word
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_frequencies = {}
        for i, (word, _) in enumerate(sorted_vocabulary):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word
            self.word_frequencies[word] = vocabulary[word]

    def vectorize(self, docs):
        def _vectorize(doc):
            vector = np.zeros(self.vocab_size)
            for word in doc.lower().split():
                if word in self.word_to_idx:
                    vector[self.word_to_idx[word]] += 1
            return vector

        return np.array([_vectorize(doc) for doc in docs])

    def save(self, path):
        with open(path, "w") as f:
            for k, v in self.word_to_idx.items():
                f.write(f"{k}\t{v}\t{self.word_frequencies[k]}\n")


class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = None
        self.classes = None
        self.word_likelihoods = None

    def fit(self, X, y):
        vocab_size = len(X[0])
        self.classes = list(set(y))
        self.class_priors = {}
        self.word_likelihoods = defaultdict(lambda: defaultdict(float))

        for c in self.classes:
            class_docs = X[y == c]  # find all docs of class c
            self.class_priors[c] = len(class_docs) / len(X)  # cal class prior
            word_count = np.sum(class_docs, axis=0) + 1  # count each word in class
            self.word_likelihoods[c] = word_count / (
                np.sum(word_count) + vocab_size
            )  # cal each word likelihood

    def predict(self, X):
        predictions = []
        for doc in X:
            log_probs = {}
            for c in self.classes:
                log_prob = np.log(self.class_priors[c]) + np.sum(
                    np.log(self.word_likelihoods[c]) * doc
                )  # calc sum(log(each word likelihood) * doc word count) + log(class prior) as probability
                log_probs[c] = log_prob
            predictions.append(max(log_probs, key=log_probs.get))
        return predictions


categories = [
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.med",
    "sci.space",
    "talk.politics.mideast",
]
# 加载20 Newsgroups数据集
train_newsgroups = fetch_20newsgroups(
    data_home="data", subset="train", categories=categories
)
test_newsgroups = fetch_20newsgroups(
    data_home="data", subset="test", categories=categories
)


def count_articles_by_category(newsgroups_data):
    category_counts = defaultdict(int)
    for target in newsgroups_data.target:
        category_name = newsgroups_data.target_names[target]
        category_counts[category_name] += 1
    return category_counts


# 输出训练集和测试集中各类文章数量
train_counts = count_articles_by_category(train_newsgroups)
test_counts = count_articles_by_category(test_newsgroups)

print("Training set article counts per category:")
for category, count in train_counts.items():
    print(f"{category}: {count}")

print("\nTest set article counts per category:")
for category, count in test_counts.items():
    print(f"{category}: {count}")

print("")

# 使用TF-IDF向量化文本数据
start_time = time.time()
vocab_size = 20000
tokenizer = Tokenizer(vocab_size)
tokenizer.load_stop_words("stop_words.txt")
tokenizer.init_vocabulary(train_newsgroups.data)
tokenizer.save("tokenizer.txt")
X_train, y_train = tokenizer.vectorize(train_newsgroups.data), train_newsgroups.target
X_test, y_test = tokenizer.vectorize(test_newsgroups.data), test_newsgroups.target

# vocab_size, accuracy, time
# 1000 0.8593 0.6098
# 5000 0.9339 1.0191
# 10000 0.9496 1.5702
# 20000 0.9521 2.4185
# 30000 0.9500 3.7728

np.set_printoptions(threshold=25, edgeitems=15, linewidth=200)
for i in range(20, 30):
    print(f"data: {X_train[i]}, label: {y_train[i]}")
print("")

# 使用朴素贝叶斯分类器
model = NaiveBayesClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 计算准确率和其他评估指标
accuracy = accuracy_score(y_test, y_pred)
print(
    f"Vocab size: {vocab_size}, Execution time: {time.time() - start_time:.4f} seconds"
)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=train_newsgroups.target_names))
