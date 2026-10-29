Heart Disease Prediction Using 1D CNN
Project Overview
This project demonstrates a Deep Learning approach to binary
classification using a 1D Convolutional Neural Network (1D-CNN) and
a heart disease dataset stored in a CSV file. It uses Python,
TensorFlow/Keras, and scikit-learn to preprocess the data, train the
model, evaluate its performance, and make a sample prediction.
Disclaimer: This is an educational project, not a medical
diagnostic tool. Model predictions should not be used to make
healthcare decisions.

Objectives
- Load and explore the heart disease CSV dataset.
- Check missing values and duplicate rows.
- Visualize the distribution of the target classes.
- Split the data into training and testing sets.
- Standardize the input features.
- Build and train a 1D-CNN model.
- Evaluate the model using test accuracy, a classification report, and
  a confusion matrix.
- Save the trained model and preprocessing scaler.
Dataset
The notebook expects a file named heart.csv at:
/content/heart.csv
The dataset is expected to contain 13 input features and a binary target
column named target.

  Feature      Description
  age        Age
  sex        Sex code used by the dataset
  cp         Chest-pain category code
  trestbps   Resting blood pressure
  chol       Serum cholesterol
  fbs        Fasting blood sugar category
  restecg    Resting ECG category
  thalach    Maximum heart rate achieved
  exang      Exercise-induced angina category
  oldpeak    ST depression value
  slope      ST-segment slope category
  ca         Number of major vessels, as coded in the dataset
  thal       Thalassemia category code
  target     Binary class label (0 or 1)

The exact meaning of category codes depends on the dataset version. The
target labels are dataset labels and should not be interpreted as a
clinical diagnosis.
Technologies Used
- Python
- Google Colab / Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- TensorFlow and Keras
- Joblib
Workflow
1. Import the required libraries.
2. Load heart.csv and inspect the data.
3. Check missing values, identify duplicate rows, and remove
   duplicates.
4. Visualize the target-class distribution.
5. Separate the input features (X) from the target (y).
6. Split the data into training and testing sets using an 80/20 split
   and stratification.
7. Fit StandardScaler on the training data and use it to transform
   both training and testing data.
8. Reshape the data to (number_of_records, 13, 1) for the 1D-CNN.
9. Build the CNN using Conv1D, MaxPooling1D, Flatten, Dense,
   and Dropout layers.
10. Train the model with early stopping.
11. Plot training and validation accuracy and loss.
12. Evaluate test performance and display a classification report and
    confusion matrix.
13. Run a sample prediction.
14. Save the model and scaler.
CNN Architecture
The notebook defines the following sequence:
- Input shape: (13, 1)
- Conv1D: 32 filters, kernel size 3, ReLU activation
- MaxPooling1D: pool size 2
- Conv1D: 64 filters, kernel size 3, ReLU activation
- Flatten
- Dense: 32 units, ReLU activation
- Dropout: 0.3
- Output Dense: 1 unit, sigmoid activation
The model is compiled with the Adam optimizer, binary cross-entropy
loss, and accuracy as a metric. Training uses up to 100 epochs, batch
size 16, a 20% validation split, and early stopping based on validation
loss.

Methodological note: The input is tabular data rather than images or
a naturally ordered time series. This CNN is used as an educational
experiment; other models may be more suitable for tabular data.
How to Run
1. Open the notebook (.ipynb) in Google Colab.
2. Upload heart.csv to the Colab session so it is available at
   /content/heart.csv.
3. Run the notebook cells from top to bottom.
4. Review the printed test metrics, classification report, and
   confusion matrix.
5. The final cell saves the model and scaler.
Saved Files
After the final notebook cell runs, these files should be created in the
Colab environment:
- heart_disease_cnn.keras --- trained Keras model.
- heart_disease_scaler.pkl --- fitted StandardScaler needed to
  preprocess future inputs.
These files are saved in /content/. Download them from Colab if you
need local copies. Keep the model and scaler together.
Evaluation
The notebook reports: - Test loss - Test accuracy - Precision, recall,
and F1-score for each target class - Confusion matrix

Actual metrics depend on the model training run. Use the values
displayed by your executed notebook; no accuracy value is assumed in
this README.
Limitations
- The dataset is relatively small, so performance estimates can vary.
- A single train/test split may not provide a complete picture of
  generalization.
- The sigmoid output score is not necessarily a calibrated medical
  probability.
- Dataset class labels and coded features may vary by dataset source.
- This model is for learning and demonstration only, not diagnosis or
  treatment.
Future Improvements
- Compare the CNN with logistic regression, Random Forest, and other
  suitable tabular-data models.
- Use cross-validation for more robust performance estimates.
- Examine precision, recall, F1-score, and class balance rather than
  accuracy alone.
- Add a user interface only after validating the preprocessing and
  model pipeline.
Author
Add your name, course, and institution here if required for your
submission.
