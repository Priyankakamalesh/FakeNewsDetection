# 📰 Fake News Detection using Machine Learning

This project applies machine learning techniques to identify fake news articles using Natural Language Processing (NLP). It classifies input news as either *Fake* or *True* based on trained models and provides manual testing functionality.

## 📁 Project Structure



FakeNewsDetection/
├── Fake.csv                # Fake news dataset
├── True.csv                # True news dataset
├── manual\_testing.csv      # Dataset reserved for manual testing
├── FakeDetectionCode.py    # Main Python script for training and testing

`

## ✅ Features

- Text preprocessing using regex and string operations
- TF-IDF vectorization of news content
- Trains and evaluates multiple ML models:
  - Logistic Regression
  - Decision Tree
  - Gradient Boosting
  - Random Forest
- Manual prediction for custom input news
- Data visualizations using Matplotlib
- Model performance evaluation via classification reports

## 🧪 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Regex (for text cleaning)
- TF-IDF Vectorizer (for feature extraction)

## ⚙ How It Works

1. *Data Merging*: Combines Fake.csv and True.csv with class labels.
2. *Cleaning*: Text is preprocessed (removes punctuation, digits, links, etc.).
3. *Feature Engineering*: Uses TfidfVectorizer to convert text to numeric form.
4. *Model Training*: Fits multiple ML models on the training set.
5. *Prediction & Evaluation*: Generates classification reports for each model.
6. *Manual Testing*: Allows user to input custom news and get predictions.

## 🧠 Models Used

| Model               | Description                |
|--------------------|----------------------------|
| Logistic Regression| Fast, linear classifier     |
| Decision Tree      | Rule-based classifier       |
| Gradient Boosting  | Boosted trees for accuracy  |
| Random Forest      | Ensemble of decision trees  |

## 📊 Visualizations

- *Bar Chart*: Number of articles per subject
- *Pie Chart*: Ratio of Fake vs True articles

## 🚀 How to Run

### 1. Clone this repository
bash
git clone https://github.com/your-username/FakeNewsDetection.git
cd FakeNewsDetection
`

### 2. Install dependencies

bash
pip install -r requirements.txt


If requirements.txt is missing, install manually:

bash
pip install pandas numpy scikit-learn matplotlib


### 3. Run the script

bash
python FakeDetectionCode.py


You’ll be prompted to enter a news article for prediction.

## 🔍 Sample Manual Input

text
Enter Your News : The Prime Minister announced new environmental policies in today's address.


*Sample Output:*


LR Prediction: TRUE News
DT Prediction: TRUE News


## 📄 Dataset Source

This project uses the [Fake and Real News Dataset from Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).

## 👩‍💻 Developed by

*Priyanka K*
Final Year Student – AI & Data Science

## 📜 License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).

---
