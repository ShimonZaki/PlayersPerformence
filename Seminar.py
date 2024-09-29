
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 5 21:48:30 2024
@author: shimo
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import  KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV, Ridge
from math import pi
import random  


# Define file paths
performance_data_path = 'PerformenceDataSet.csv'
player_details_path = 'PlayersinfoDataSet.csv'

def normalize_dataset_in_memory(data, exclude_columns):
    # Separate the columns to be excluded from the normalization process
    excluded_data = data[exclude_columns]
    
    # Select only the numerical features for normalization, excluding the specified columns
    numerical_features = data.drop(columns=exclude_columns).select_dtypes(include=['float64', 'int64'])

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the numerical features and transform them
    normalized_data = scaler.fit_transform(numerical_features)

    # Convert the normalized data back to a DataFrame
    normalized_df = pd.DataFrame(normalized_data, columns=numerical_features.columns, index=data.index)

    # Concatenate the excluded columns back with the normalized data
    final_df = pd.concat([excluded_data, normalized_df], axis=1)

    return final_df

# Function to get the most effective features for goalscoreing.
def perform_ridge_regression(data):
    # Exclude features that are directly related to goals and 'id' as it is not a feature
    features_to_exclude = [
        'goals_from_inside_box', 'right_foot_goals', 'left_foot_goals',
        'headed_goals', 'goals_from_outside_box', 'home_goals', 'away_goals',
        'id','goals','winning_goal'
    ]
    X = data.drop(columns=features_to_exclude, errors='ignore')
    y = data['goals'] 

    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_columns]

    # Impute missing values and scale features
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Perform Ridge regression with cross-validation to determine the best alpha
    alphas = np.logspace(-6, 6, 13)
    ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
    ridge_cv.fit(X_scaled, y)

    # The best alpha and coefficients
    best_alpha = ridge_cv.alpha_
    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(X_scaled, y)

    # Calculate R^2 score
    r_squared = ridge_model.score(X_scaled, y)

    # Create a Series for feature importances
    feature_importance = pd.Series(ridge_model.coef_, index=numeric_columns).sort_values(key=abs, ascending=False)

    return ridge_model, feature_importance, r_squared

# Function to generate radar charts for players
def generate_radar_for_player(player_id, data, player_details_data):
    player_data = data[data['id'] == player_id]
    if player_data.empty:
        print("Player ID not found in the dataset.")
        return
    
    # Find player's name using the player details DataFrame
    player_name = player_details_data[player_details_data['id'] == player_id]['name'].iloc[0] if not player_details_data[player_details_data['id'] == player_id].empty else 'Unknown'

    position = player_data['position'].iloc[0]
    if position == 'Goalkeeper':
        metrics = ['catches', 'saves_from_penalty', 'successful_launches', 'saves_made_from_outside_box', 'saves_made_from_inside_box', 'gk_successful_distribution']
    elif position == 'Defender':
        metrics = ['total_clearances', 'blocks', 'interceptions', 'duels_won', 'tackles_won', 'recoveries']
    elif position == 'Midfielder':
        metrics = ['successful_long_passes', 'successful_passes_opposition_half', 'successful_short_passes', 'key_passes_attempt_assists',
                   'forward_passes','key_passes_attempt_assists','interceptions']
    elif position == 'Forward':
        metrics = ['total_touches_in_opposition_box', 'shots_on_target', 'headed_goals', 'successful_dribbles', 'aerial_duels_won']
    else:
        print("Position not recognized.")
        return
    
    player_metrics = player_data[metrics].iloc[0].fillna(0).tolist()
    player_metrics += player_metrics[:1]
    
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1], metrics)
    ax.set_yticklabels([])
    ax.plot(angles, player_metrics, color='g', linewidth=2, linestyle='solid')
    ax.fill(angles, player_metrics, color='b', alpha=0.1)
    ax.set_title(f'{position} Performance: {player_name} (ID {player_id})', size=15, color='b', y=1.1)
    
    plt.show()


if os.path.exists(performance_data_path) and os.path.exists(player_details_path):
    # Load datasets
    performance_data = pd.read_csv(performance_data_path)
    player_details = pd.read_csv(player_details_path)
    
    # Columns to exclude normalization
    exclude_columns = ['id', 'position']

    # Normalize performance 
    normalized_performance_data = normalize_dataset_in_memory(performance_data, exclude_columns)

    # the sample id to display
    sample_id = 6130

  # Use an actual player ID from your dataset
    generate_radar_for_player(sample_id, normalized_performance_data, player_details)
    
    
    # Perform Ridge regression and R^2 score
    ridge_model, feature_importance, r_squared = perform_ridge_regression(normalized_performance_data)

    # Display the R^2 score and top contributing factors
    print(f'R^2 score: {r_squared}')
    print("Top contributing factors to players' goals:")
    print(feature_importance.head(10))

    # Plot the top contributing factors
    plt.figure(figsize=(10, 6))
    feature_importance.head(10).plot(kind='barh')
    plt.gca().invert_yaxis()  # To display the most important feature at the top
    plt.title('Top Contributing Factors to Players\' Goals')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.show()
        
    
else:
    print(f"Files not found: {performance_data_path} or {player_details_path}")
    

if os.path.exists(performance_data_path) and os.path.exists(player_details_path):
    try:
        # Load the datasets
        normalized_performance_data = pd.read_csv(performance_data_path)
        player_details = pd.read_csv(player_details_path)

        # Define features and target
        features = ['total_shots', 'penalties_taken', 'shots_on_target', 'total_touches_in_opposition_box', 'goal_assists', 'time_played', 'successful_dribbles']
        X = normalized_performance_data[features]
        y = normalized_performance_data['goals']

        # K-Fold Cross-Validation setup
        kf = KFold(n_splits=12, shuffle=True, random_state=0)
        # Models to evaluate
        models = {
            'RandomForestRegressor': RandomForestRegressor(n_estimators=250, random_state=50),
            'LassoRegression': Lasso(random_state=50),
            'ElasticNet': ElasticNet(random_state=50),
            'XGBRegressor': XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=50)  # XGBoost regressor
        }

        # Dictionary to store the scores for each model
        model_scores = {model_name: {'MSE': [], 'RMSE': [], 'R2': []} for model_name in models.keys()}

        # Store individual predictions
        all_predictions = []

        # Cross-validation loop
        for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Calculate metrics
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, predictions)

                # Store the scores
                model_scores[model_name]['MSE'].append(mse)
                model_scores[model_name]['RMSE'].append(rmse)
                model_scores[model_name]['R2'].append(r2)

                # Store individual predictions only if predicted goals > 5
                for i, index in enumerate(test_index):
                    if predictions[i] > 5:  # Check if the predicted goals are more than 5
                        player_id = normalized_performance_data.iloc[index]['id']
                        actual_goals = y_test.iloc[i]
                        all_predictions.append((player_id, predictions[i], actual_goals))
                        
                 # Store predictions and actual goals for visualization
                if model_name == 'RandomForestRegressor':
                    model_scores[model_name].setdefault('predictions', []).extend(predictions.tolist())
                    model_scores[model_name].setdefault('actual_goals', []).extend(y_test.tolist())   
                            
        # Ensure uniqueness in the random 10 predictions
        presented_player_ids = set()
        random_10_predictions = []
        while len(random_10_predictions) < 10 and all_predictions:
            prediction = random.choice(all_predictions)
            if prediction[0] not in presented_player_ids:
                random_10_predictions.append(prediction)
                presented_player_ids.add(prediction[0])
            all_predictions.remove(prediction)  # Remove the chosen prediction to avoid retrying the same prediction

        # Join with player details to get player names
        random_10_df = pd.DataFrame(random_10_predictions, columns=['id', 'Predicted Goals', 'Actual Goals'])
        random_10_df = random_10_df.merge(player_details[['id', 'name']], on='id', how='left')

        # Print random 10 unique predictions with player names and actual goals
        print("\nRandom 10 Unique Predictions with More Than 5 Predicted Goals:")
        for _, row in random_10_df.iterrows():
            print(f"Player: {row['name']}, Predicted Goals: {row['Predicted Goals']}, Actual Goals: {row['Actual Goals']}")
        
        # Calculate and print the average score for each model across all folds
        for model_name, scores in model_scores.items():
            avg_mse = np.mean(scores['MSE'])
            avg_rmse = np.mean(scores['RMSE'])
            avg_r2 = np.mean(scores['R2'])
          
            print(f"\nModel: {model_name}")
            print(f"Average MSE: {avg_mse}")
            print(f"Average RMSE: {avg_rmse}")
            print(f"Average R^2: {avg_r2}")    
            
             # Visualizations
        # 1. Model Performance Comparison
        plt.figure(figsize=(10, 6))
        for metric in ['MSE', 'RMSE', 'R2']:
            avg_scores = [np.mean(model_scores[model][metric]) for model in models.keys()]
            sns.barplot(x=list(models.keys()), y=avg_scores)
            plt.title(f'Average {metric} for Each Model')
            plt.ylabel(metric)
            plt.xlabel('Model')
            plt.xticks(rotation=45)
            plt.show()
    
      # Predictions vs. Actual Goals for RandomForestRegressor
        predicted_goals = model_scores['RandomForestRegressor']['predictions']
        actual_goals = model_scores['RandomForestRegressor']['actual_goals']
        
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(predicted_goals)), predicted_goals, color='r', alpha=0.5, label='Predicted')
        plt.scatter(range(len(actual_goals)), actual_goals, color='g', alpha=0.5, label='Actual')
        plt.title('Predicted vs. Actual Goals for RandomForestRegressor')
        plt.xlabel('Sample Index')
        plt.ylabel('Goals')
        plt.legend()
        plt.show()
        
        # Error Distribution for RandomForestRegressor
        errors = np.array(predicted_goals) - np.array(actual_goals)
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True, color='m')
        plt.title('Distribution of Prediction Errors for RandomForestRegressor')
        plt.xlabel('Error (Predicted Goals - Actual Goals)')
        plt.ylabel('Frequency')
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"Files not found: {performance_data_path} or {player_details_path}")
