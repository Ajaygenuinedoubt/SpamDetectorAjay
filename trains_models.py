# Import necessary libraries
import pandas as pd
import zipfile
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
from flask import Flask, request, jsonify

# Load and prepare the dataset
def load_sms_spam_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    response = requests.get(url)
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        with z.open('SMSSpamCollection') as file:
            df = pd.read_csv(file, sep='\t', header=None, names=['label', 'message'])
    return df

# Preprocess the text data
def preprocess_text(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['message'])
    y = data['label'].map({'ham': 0, 'spam': 1})
    return X, y, vectorizer

# Load and preprocess the dataset
df = load_sms_spam_data()
X, y, vectorizer = preprocess_text(df)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

# Train all classifiers and save them to .pkl files
for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    print(f"{name} training complete.")
    # Save each model as a .pkl file
    with open(f"{name}_model.pkl", 'wb') as model_file:
        pickle.dump(clf, model_file)

# Save the vectorizer to a .pkl file
with open('tfidf_vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

# FLASK APP
app = Flask(__name__)

# Load models and vectorizer
models = {}
for model_name in classifiers.keys():
    with open(f'{model_name}_model.pkl', 'rb') as model_file:
        models[model_name] = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to classify a given message as 'spam' or 'ham'.
    Request should contain JSON with 'message' and 'model' (RandomForest, GradientBoosting, LogisticRegression, SVM, AdaBoost).
    Example request body: {"message": "Your free lottery!", "model": "RandomForest"}
    """
    data = request.json
    message = data.get('message', '')
    model_name = data.get('model', 'RandomForest')
    
    if model_name not in models:
        return jsonify({"error": "Model not found. Available models: RandomForest, GradientBoosting, LogisticRegression, SVM, AdaBoost"}), 400
    
    # Transform the message using the vectorizer
    X_new = vectorizer.transform([message])
    
    # Predict using the selected model
    model = models[model_name]
    y_pred = model.predict(X_new)
    
    # Return the prediction result
    result = 'spam' if y_pred[0] == 1 else 'ham'
    return jsonify({"message": message, "prediction": result})

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
