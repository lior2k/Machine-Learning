import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
import contractions
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

data_file = open("C:\\Users\\lior2\\OneDrive\\Desktop\\dataset\\yelp_academic_dataset_review.json", encoding='utf-8')
data = []
i = 0
num_reviews = 10000
for line in data_file:
    data.append(json.loads(line))
    i += 1
    if i == num_reviews:
        break

data_file.close()

reviews_list = []
label_list = []
# countPositive, countNeutral, countNegative = 0, 0, 0
for d in data:
    expanded_words = []
    for word in d["text"].split():
        expanded_words.append(contractions.fix(word))
    reviews_list.append(' '.join(expanded_words))
    label = 0
    if d['stars'] == 1 or d['stars'] == 2:
        label = -1
        # countNegative += 1
    elif d['stars'] == 4 or d['stars'] == 5:
        label = 1
        # countPositive += 1
    # else:
    #     countNeutral += 1
    label_list.append(label)
# print(countPositive, countNeutral, countNegative)

X_train, X_test, y_train, y_test = train_test_split(reviews_list, label_list, test_size=0.2, random_state=0)

tfidf = TfidfVectorizer(min_df=300, ngram_range=(1, 3))
matrix = tfidf.fit_transform(X_train)
print(len(matrix.toarray()))
print(len(matrix.toarray()[0]))
print(len(matrix.toarray()[1]))

# model = svm.SVC(kernel='linear')
model = KNeighborsClassifier(n_neighbors=7)
# model = RandomForestClassifier(n_estimators=100, random_state=0)
# model = LogisticRegression(random_state=0, max_iter=1000)

model.fit(matrix, y_train)

matrix_test = tfidf.transform(X_test)
y_pred = model.predict(matrix_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("F1: ", metrics.f1_score(y_test, y_pred, average='weighted'))
print("Recall: ", metrics.recall_score(y_test, y_pred, average='weighted'))
