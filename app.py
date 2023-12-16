from flask import Flask, render_template, request
import os
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Configure generative AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# FAQ context
context = '''Question: Can you explain the difference between supervised and unsupervised learning?

Answer: Supervised learning involves training a model on a labeled dataset, while unsupervised learning deals with unlabeled data, aiming to find patterns and relationships without predefined outcomes.
Question: How do you handle missing data in a dataset?

Answer: I assess the extent of missing data and choose an appropriate strategy, such as imputation methods based on means, medians, or using advanced techniques like regression or machine learning algorithms.
Question: What is the curse of dimensionality, and how does it affect model performance?

Answer: The curse of dimensionality refers to challenges that arise when working with high-dimensional data. It can lead to increased model complexity, overfitting, and reduced predictive performance. Feature selection and dimensionality reduction techniques can help mitigate these issues.
Question: Explain the concept of regularization in machine learning.

Answer: Regularization is a technique used to prevent overfitting by adding a penalty term to the model's cost function. It discourages the model from assigning excessive importance to any single feature, promoting a more generalized solution.
Question: What is the purpose of cross-validation, and how does it work?

Answer: Cross-validation assesses a model's performance by splitting the dataset into multiple subsets, training the model on some and testing on others. This helps to ensure that the model generalizes well to unseen data and provides a more robust evaluation.
Question: Can you explain the bias-variance tradeoff?

Answer: The bias-variance tradeoff represents the balance between a model's ability to fit the training data (low bias) and generalize to new, unseen data (low variance). Achieving the optimal tradeoff is crucial for model performance.
Question: How do decision trees work, and what are their limitations?

Answer: Decision trees recursively split data based on features to make decisions. While they are interpretable and easy to understand, they can be prone to overfitting. Techniques like pruning and ensemble methods address these limitations.
Question: What is feature engineering, and why is it important?

Answer: Feature engineering involves transforming raw data into meaningful features that enhance a model's performance. It is crucial as well-engineered features can improve a model's ability to capture patterns and relationships in the data.
Question: Explain the difference between bagging and boosting.

Answer: Bagging (Bootstrap Aggregating) involves training multiple models independently on different subsets of the data and combining their predictions. Boosting, on the other hand, builds models sequentially, focusing on correcting errors made by previous models.
Question: How does the k-nearest neighbors (KNN) algorithm work?

Answer: KNN classifies data points based on the majority class of their k-nearest neighbors in feature space. It is a non-parametric and lazy learning algorithm, meaning it doesn't make assumptions about the underlying data distribution until prediction time.
Question: What is the difference between L1 and L2 regularization?

Answer: L1 regularization adds the absolute values of the coefficients to the cost function, encouraging sparsity in feature selection. L2 regularization adds the squared values of the coefficients, penalizing large coefficients and promoting smoother models.
Question: How would you handle imbalanced datasets in classification problems?

Answer: Imbalanced datasets can be addressed by techniques such as oversampling the minority class, undersampling the majority class, or using algorithms designed to handle class imbalance, such as Synthetic Minority Over-sampling Technique (SMOTE).
Question: Explain the concept of A/B testing.

Answer: A/B testing is a method to compare two versions (A and B) of a variable, typically by testing a subject's response to variant A against variant B, and determining which of the two performs better.
Question: What is the purpose of a confusion matrix in classification?

Answer: A confusion matrix provides a summary of a model's performance, showing the number of true positive, true negative, false positive, and false negative predictions. It is a valuable tool for evaluating the effectiveness of a classification model.
Question: How does gradient descent work in the context of machine learning?

Answer: Gradient descent is an optimization algorithm used to minimize the cost function by iteratively adjusting model parameters in the direction of the steepest decrease in the cost. It is a key component in training machine learning models.
Question: What is the difference between batch gradient descent and stochastic gradient descent?

Answer: Batch gradient descent computes the gradient of the entire dataset before updating model parameters, while stochastic gradient descent updates parameters after computing the gradient for each individual data point. Mini-batch gradient descent is a compromise between the two, using a small subset of the data for each iteration.
Question: How do you handle outliers in a dataset?

Answer: Outliers can be addressed by techniques such as removing them, transforming the data, or using robust statistical methods. The approach depends on the nature of the data and the impact outliers have on the model.
Question: What is the purpose of principal component analysis (PCA)?

Answer: PCA is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving the most important information. It is often used to reduce multicollinearity and improve model performance.
Question: Explain the concept of overfitting and how to prevent it.

Answer: Overfitting occurs when a model fits the training data too closely, capturing noise rather than underlying patterns. Techniques to prevent overfitting include regularization, cross-validation, and using more training data.
Question: How would you approach a time-series forecasting problem?

Answer: Time-series forecasting involves understanding temporal patterns. I would start by analyzing trends, seasonality, and autocorrelation, and then use appropriate models such as ARIMA, Exponential Smoothing, or machine learning algorithms tailored for time-series data.
Question: What is deep learning, and how does it differ from traditional machine learning?

Answer: Deep learning is a subset of machine learning that involves neural networks with multiple layers (deep neural networks). It excels at learning hierarchical representations and is particularly effective for tasks like image and speech recognition. Traditional machine learning typically involves simpler models with fewer layers.
Question: How do you assess model performance in regression problems?

Answer: In regression, common metrics for assessing model performance include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared. The choice of metric depends on the specific requirements of the problem.
Question: Can you explain the concept of ensemble learning?

Answer: Ensemble learning combines predictions from multiple models to improve overall performance. Common ensemble methods include Random Forests (bagging) and Gradient Boosting (boosting).'''

# Route for the FAQ page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle user input and generate response
@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        user_input = request.form['user_input']
        response = model.generate_content(f"{context} - {user_input}")
        return render_template('index.html', user_input=user_input, response=response.text)

if __name__ == '__main__':
    app.run(debug=True)
