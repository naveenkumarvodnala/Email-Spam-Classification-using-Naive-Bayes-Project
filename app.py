from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the trained model and vectorizer
with open('Email_Span.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input email from the form
    email = request.form['email']

    # Vectorize the email
    email_vectorized = vectorizer.transform([email])

    # Predict using the Naive Bayes model
    prediction = model.predict(email_vectorized)

    # Map the result to labels
    result = 'Spam' if prediction[0] == 1 else 'Not Spam'

    # Return the result back to the HTML page
    return render_template('index.html', prediction_text=f'The email is: {result}')

if __name__ == '__main__':
    app.run(debug=True)
