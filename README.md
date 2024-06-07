Diabetes Prediction Web Application
Overview
This project aims to develop a web application that predicts the type of diabetes (Type 1, Type 2, or No diabetes) based on user-reported symptoms. The application is built using Flask, a lightweight Python web framework, and utilizes a Decision Tree Classifier model for prediction. The model is evaluated using cross-validation to ensure its reliability and accuracy.

Table of Contents
Overview
Features
Installation
Usage
Dataset
Model Training
Web Application
Results and Discussion
Contributing
License
References
Features
User-friendly web interface for inputting symptoms.
Predicts the type of diabetes based on symptoms.
Stores user inputs and predictions in a SQLite database.
Evaluates model performance using cross-validation.
Displays cross-validation scores.
Installation
Clone the repository:
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install dependencies:
pip install -r requirements.txt
Run the application:
python app.py
sage
Open your web browser and go to http://127.0.0.1:5000/.
Enter your name, age, and select the symptoms you are experiencing.
Click the "Submit" button to get the prediction.
The result page will display the predicted type of diabetes and the model's accuracy.
Dataset
The dataset consists of 16 features representing common symptoms of diabetes and a label indicating the type of diabetes. The features are binary, where 1 indicates the presence of a symptom and 0 indicates the absence. The data is organized in the form of a dictionary and then converted into a DataFrame using the pandas library for easier processing.
