from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
import os
print("Current Working Directory:", os.getcwd())
print("Files in directory:", os.listdir())



app = Flask(__name__)

# Load trained model
model = joblib.load("floods.save")


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict')
def predict_page():
    return render_template("predict.html")


@app.route('/result', methods=['POST'])
def result():
    try:
        # Get form inputs
        cloud = float(request.form['cloud'])
        annual = float(request.form['annual'])
        jan_feb = float(request.form['jan_feb'])
        mar_may = float(request.form['mar_may'])
        jun_sep = float(request.form['jun_sep'])

        # Arrange input in correct order
        input_data = pd.DataFrame({
            'Cloud Cover': [cloud],
            'ANNUAL': [annual],
            'Jan-Feb': [jan_feb],
            'Mar-May': [mar_may],
            'Jun-Sep': [jun_sep]
        })
        # Prediction
        prediction = model.predict(input_data)[0]

        # Optional: probability (if model supports it)
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            output = f"Possibility of Severe Flood)"
            color = "red"
        else:
            output = f"No Possibility of Severe Flood"
            color = "green"

        return render_template("result.html",
                               prediction=output,
                               color=color)

    except Exception as e:
        return render_template("result.html",
                               prediction="Invalid Input. Please enter numeric values only.",
                               color="orange")


if __name__ == "__main__":
    app.run(debug=True)
