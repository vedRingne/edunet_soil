Soil Fertility Prediction using Machine Learning
Project Overview
Soil fertility plays a crucial role in agricultural productivity. This project builds a machine learning model to predict soil fertility based on chemical properties such as pH, Nitrogen, Phosphorus, Potassium, and Organic Matter content.

Dataset
The dataset contains various soil parameters and fertility labels. It is used to train a model that classifies soil samples into fertile or infertile categories.

Installation & Setup
Clone the repository:
bash
Copy
Edit
git clone https://github.com/vedRingne/edunet_soil.git
cd soil-fertility-prediction
Install required libraries:
bash
Copy
Edit
pip install -r requirements.txt
Run the Jupyter Notebook:
bash
Copy
Edit
jupyter notebook soil_fertility_prediction.ipynb
Model & Approach
Preprocessing: Handling missing values, encoding categorical features, normalizing data
Model Used: Random Forest Classifier (can be replaced with other ML models)
Evaluation Metrics: Accuracy, Precision, Recall, F1-score
Usage
Load the dataset and preprocess it
Train the model on the dataset
Evaluate and interpret the results
Predict soil fertility for new samples
Example Usage
python
Copy
Edit
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make a prediction
new_sample = [[6.5, 50, 30, 20, 3.5]]  # Replace with real values
prediction = model.predict(new_sample)
print("Predicted Fertility:", prediction)
Results
Model achieved XX% accuracy on the test set
Feature importance analysis shows pH and Nitrogen as key factors
Future Improvements
Hyperparameter tuning for better accuracy
Experimenting with deep learning models
Building a web app for user interaction
📝 Contributing
Feel free to fork this repository and submit a pull request for improvements.

📜 License
This project is licensed under the MIT License.

