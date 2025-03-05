from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load models and vectorizer from the .pkl files
models = {}
model_names = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM', 'AdaBoost']

for model_name in model_names:
    with open(f'{model_name}_model.pkl', 'rb') as model_file:
        models[model_name] = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route('/')
def index():
    """
    Renders the home page with a form to input message and select multiple classifiers.
    """
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles form submission, predicts whether the message is spam or ham using the selected models, and renders the results.
    """
    message = request.form.get('message')
    selected_models = request.form.getlist('models')  # Retrieve the list of selected models

    if not message or not selected_models:
        return render_template('index.html', model_names=model_names, error="Please provide a message and select at least one model.")

    # Transform the input message using the vectorizer
    X_new = vectorizer.transform([message])

    # Store the results for each model
    results = {}
    for model_name in selected_models:
        if model_name in models:
            model = models[model_name]
            y_pred = model.predict(X_new)
            # Determine the result (spam or ham)
            result = 'spam' if y_pred[0] == 1 else 'ham'
            results[model_name] = result

    return render_template('index.html', model_names=model_names, results=results, message=message, selected_models=selected_models)

if __name__ == '__main__':
    app.run(debug=True)
