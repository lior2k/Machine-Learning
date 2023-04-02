from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

"""
Restaurant
Shopping
Health & Medical

              Restaurant | Shopping | Health & Medical
 Business #1 |     1     |    0     |     1           
 Business #2 |     0     |    0     |     1           
    ...          ...         ...         ...         
"""


def map_categories_to_labels(st: str) -> [int]:
    result = [0, 0, 0]
    if "Restaurant" in st:
        result[0] = 1
    if "Shopping" in st:
        result[1] = 1
    if "Health & Medical" in st:
        result[2] = 1
    return result


data = {}

business_file = open("C:\\Users\\lior2\\OneDrive\\Desktop\\dataset\\yelp_academic_dataset_business.json",
                     encoding='utf-8')
for line in business_file:
    current_business = json.loads(line)
    if current_business["review_count"] < 60 or current_business["review_count"] > 80:
        continue
    if current_business['categories'] is None:
        continue
    labels = map_categories_to_labels(current_business['categories'])
    if labels == [0, 0, 0]:  # disregard business that are non of our options (medical / restaurant / shopping)
        continue
    data[current_business['business_id']] = [[], labels]
business_file.close()

reviews_file = open("C:\\Users\\lior2\\OneDrive\\Desktop\\dataset\\yelp_academic_dataset_review.json", encoding='utf-8')
for line in reviews_file:
    current_review = json.loads(line)
    business_id = current_review['business_id']
    if data.get(business_id) is None:
        continue
    text = current_review['text']
    data[business_id][0].append(text)
reviews_file.close()

reviews = []
labels = []
for key in data.keys():
    reviews.append("".join(data[key][0]))
    labels.append(data[key][1])

X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=0)

tfidf = TfidfVectorizer(min_df=300, stop_words='english', ngram_range=(1, 3))
matrix = tfidf.fit_transform(X_train)

# model = MultiOutputClassifier(svm.SVC(kernel='linear'))
# model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5))
# model = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=0))
model = MultiOutputClassifier(LogisticRegression(random_state=0, max_iter=1000))

model.fit(matrix, y_train)

matrix_test = tfidf.transform(X_test)
y_pred = model.predict(matrix_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))










