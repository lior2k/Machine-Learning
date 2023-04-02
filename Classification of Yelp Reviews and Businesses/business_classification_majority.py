from sklearn import metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import json
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

"""
Restaurant
Shopping
Health & Medical
Arts & Entertainment

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


# extract businesses from dataset and split them 80% 20%
train_data = {}
test_data = {}
num_businesses = 4489
threshold = num_businesses * 0.8
i = 0
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
    if i < threshold:
        train_data[current_business['business_id']] = [[], labels]
    else:
        test_data[current_business['business_id']] = [[], labels]
    i += 1
business_file.close()

# extract reviews from dataset
reviews_file = open("C:\\Users\\lior2\\OneDrive\\Desktop\\dataset\\yelp_academic_dataset_review.json", encoding='utf-8')
for line in reviews_file:
    current_review = json.loads(line)
    business_id = current_review['business_id']
    if train_data.get(business_id) is None and test_data.get(business_id) is None:
        continue
    text = current_review['text']
    if train_data.get(business_id) is not None:
        train_data[business_id][0].append(text)
    else:
        test_data[business_id][0].append(text)
reviews_file.close()

# test train split
business_reviews_train = [review for business_id in train_data for review in train_data[business_id][0]]
labels_train = [train_data[business_id][1] for business_id in train_data for review in train_data[business_id][0]]

business_reviews_test = [test_data[business_id][0] for business_id in test_data]
labels_test = [test_data[business_id][1] for business_id in test_data]

# vectorize the reviews of the training sample
tfidf = TfidfVectorizer(min_df=300, stop_words='english', ngram_range=(1, 3))
matrix = tfidf.fit_transform(business_reviews_train)
print(len(matrix.toarray()))
print(len(matrix.toarray()[0]))

# create and train the model on the sample
model = MultiOutputClassifier(LogisticRegression(random_state=0, max_iter=1000))
model.fit(matrix, labels_train)

# test the model on the test set
# for each business, run the model on all the business reviews and predict their label, then for each category
# take the majority vote for that category, for example if we have 10 reviews and 6 of those were labeled restaurant,
# business will be classified as a restaurant, same for other categories.
prediction = []
for i in range(len(business_reviews_test)):
    current_business = business_reviews_test[i]
    matrix = tfidf.transform(current_business)
    y_pred = model.predict(matrix)
    majority_vote = [1 if sum(col) > len(y_pred) / 2 else 0 for col in zip(*y_pred)]
    prediction.append(majority_vote)

test_pred = []
for j in range(len(labels_test)):
    test_pred.append([1, 0, 0])
print("Accuracy:", metrics.accuracy_score(labels_test, prediction))
print("test: ", metrics.accuracy_score(test_pred, prediction))
