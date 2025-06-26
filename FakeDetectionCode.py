import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as ttp
from sklearn.metrics import classification_report
import re  
import string
import matplotlib.pyplot as plt  

# Read datasets
path1 = "True.csv"
path2 = "Fake.csv"
data_true = pd.read_csv(path1)
data_fake = pd.read_csv(path2)

# Add class labels
data_true["class"] = 1
data_fake["class"] = 0

# Manual testing data (last 10 rows from each)
data_true_manual_testing = data_true.tail(10)
for i in range(21416, 21406, -1):
    data_true.drop([i], axis=0, inplace=True)

data_fake_manual_testing = data_fake.tail(10)
for i in range(23480, 23470, -1):  # Assuming 23480 is the last index in data_fake
    data_fake.drop([i], axis=0, inplace=True)

# Combine manual testing data
data_manual_testing = pd.concat([data_fake_manual_testing, data_true_manual_testing], axis=0)
data_manual_testing.to_csv("manual_testing.csv", index=False)

# Merge datasets
data_merge = pd.concat([data_fake, data_true], axis=0)

# Plot articles per subject
print(data_merge.groupby(['subject'])['text'].count())
print()
data_merge.groupby(['subject'])['text'].count().plot(kind="bar")
plt.title("Articles per subject", size=20)
plt.xlabel("Category", size=20)
plt.ylabel("Article count", size=20)
plt.show()

# Plot fake vs true news
print(data_merge.groupby(['class'])['text'].count())
print("0 = Fake news\n1 = True news")
data_merge.groupby(['class'])['text'].count().plot(kind="pie", autopct="%1.1f%%", labels=["Fake", "True"])
plt.title("Fake news and True News", size=20)
plt.show()
print()

# Drop unneeded columns
data = data_merge.drop(["title", "subject", "date"], axis=1)

# Shuffle the dataset
data = data.sample(frac=1)

# Check for null values
data.isnull().sum()

# Text preprocessing function
def filtering(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply preprocessing
data["text"] = data["text"].apply(filtering)

# Features and labels
x = data["text"]
y = data["class"]

# Train-test split
x_train, x_test, y_train, y_test = ttp(x, y, test_size=0.25, random_state=0)

# TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer()
xv_train = vector.fit_transform(x_train)
xv_test = vector.transform(x_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression 
LR = LogisticRegression() 
LR.fit(xv_train, y_train) 
print("Logistic Regression Accuracy:", LR.score(xv_test, y_test).round(2))
pred_LR = LR.predict(xv_test) 
print("Logistic Regression Report:\n", classification_report(y_test, pred_LR))
print()

# Decision Tree
from sklearn.tree import DecisionTreeClassifier 
DT = DecisionTreeClassifier() 
DT.fit(xv_train, y_train) 
print("Decision Tree Accuracy:", DT.score(xv_test, y_test).round(2))
pred_DT = DT.predict(xv_test) 
print("Decision Tree Report:\n", classification_report(y_test, pred_DT))
print()

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier 
GBC = GradientBoostingClassifier(random_state=0) 
GBC.fit(xv_train, y_train) 
print("Gradient Boosting Accuracy:", GBC.score(xv_test, y_test).round(2))
pred_GBC = GBC.predict(xv_test) 
print("Gradient Boosting Report:\n", classification_report(y_test, pred_GBC))
print()

# Random Forest
from sklearn.ensemble import RandomForestClassifier 
RFC = RandomForestClassifier(random_state=0) 
RFC.fit(xv_train, y_train) 
print("Random Forest Accuracy:", RFC.score(xv_test, y_test).round(2))
pred_RFC = RFC.predict(xv_test) 
print("Random Forest Report:\n", classification_report(y_test, pred_RFC))
print()

# Output label
def output_label(n): 
    if n == 0: 
        return "FAKE News"     
    elif n == 1: 
        return "TRUE News"     

# Manual testing function
def manual_testing(news):     
    testing_news = {"text": [news]}     
    new_def_test = pd.DataFrame(testing_news)     
    new_def_test["text"] = new_def_test["text"].apply(filtering)      
    new_x_test = new_def_test["text"]    
    new_xv_test = vector.transform(new_x_test)     
    
    pred_LR = LR.predict(new_xv_test)     
    pred_DT = DT.predict(new_xv_test) 
      
    return print("\n\nLR Prediction: {} \nDT Prediction: {}".format(
        output_label(pred_LR[0]), output_label(pred_DT[0])))

# Run manual test
news = str(input("Enter Your News : ")) 
manual_testing(news)

