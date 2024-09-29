
Soccer Player Performance Analysis and Prediction
This project aims to analyze and predict soccer player performance using various machine learning techniques.
It processes player data, performs predictive modeling, and visualizes important metrics through several advanced methods.

Key Features:
Data Normalization:
The dataset is normalized using MinMaxScaler, excluding specific non-numerical columns
(such as player ID and position), to ensure the features are scaled appropriately for machine learning algorithms.

Ridge Regression for Feature Importance:
Ridge regression is used to identify the most significant features contributing to player goals.
Cross-validation is applied to determine the optimal alpha for the Ridge model, ensuring the best regularization strength.

Radar Charts for Player Visualization:
The code generates radar charts to represent individual player metrics based on their position (e.g., goalkeeper, defender, midfielder, forward).
These charts provide a visual comparison of different performance aspects, such as saves, passes, or shots.

Model Evaluation:
Several machine learning models are employed, including RandomForestRegressor, Lasso, ElasticNet, and XGBRegressor.
The models are evaluated using K-Fold cross-validation, and performance metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²) are calculated for each model.

Predictions and Visualizations:
The code compares predicted and actual goals for each player, particularly focusing on players with more than five predicted goals.
Visualizations such as bar charts, scatter plots, and error distribution histograms are generated to illustrate model performance and prediction accuracy.

Cross-Validation and Predictions:
The models undergo 12-fold cross-validation, and predictions are stored for further analysis.
The code also extracts and displays a random selection of players with high predicted goal scores for a closer evaluation.

