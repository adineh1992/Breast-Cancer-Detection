Breast Cancer Detection Using Machine Learning Classifier 

Overview

This Jupyter notebook provides an end-to-end solution for the detection of breast cancer using supervised machine learning classifiers. It explores the classification of tumor cells into malignant or benign categories based on extracted cell features.

Libraries and Data

Libraries Used: Pandas, Seaborn, Matplotlib, NumPy, Scikit-learn, Imbalanced-learn.

Data Source: Breast cancer dataset from Scikit-learn.

Models and Evaluation

Several classifiers were tested, including Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest. Each model's performance was evaluated using accuracy scores, precision, recall, and F1 scores to determine the most effective classifier for this specific problem.

Best Model and Optimization

The Random Forest classifier emerged as the best model, achieving the highest accuracy. To further enhance the model's performance, GridSearchCV was employed to tune the hyperparameters, leading to an improved accuracy score. This process involved systematically testing a range of hyperparameters to find the optimal values that result in the best model performance.

After applying GridSearchCV, the accuracy score of the best model was found to be 95.81%. Furthermore, an ROC (Receiver Operating Characteristic) curve was plotted to evaluate the model's performance visually. 

This optimized Random Forest model, with its fine-tuned hyperparameters, demonstrates a robust capability in detecting breast cancer from the dataset features, providing a reliable tool for predictive analysis in medical diagnostics.

