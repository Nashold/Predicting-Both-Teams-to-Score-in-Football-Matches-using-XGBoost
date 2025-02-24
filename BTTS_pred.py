import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('PL_data.csv', encoding='latin-1')
data = data.dropna()  

numerical_features = data.select_dtypes(include=['number']).columns
numerical_data = data[numerical_features]

# Now apply StandardScaler to the numerical data only
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# If you need to have the scaled data back in the original DataFrame:
data[numerical_features] = scaled_data


data['GF'] = pd.to_numeric(data['GF'], errors='coerce')
data['GA'] = pd.to_numeric(data['GA'], errors='coerce')

# Create a new column for # 'Both Teams to Score' (if both teams have scored goals, 1, if not 0)
data['BTTS'] = data.apply(lambda row: 1 if row['GF'] > 0 and row['GA'] > 0 else 0, axis=1)


data = data.dropna(subset=['BTTS'])


print(data.head())

# Select Features and Target Variable
X = data[['xG', 'xGA', 'Poss','Sh', 'SoT', 'Dist', 'FK', 'PK']]  # Features
y = data['BTTS']  # Target

# Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the XGBoost Model and Hyperparameter Grid
xgb_model = xgb.XGBClassifier(use_label_encoder=False)

# Define the hyperparameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],           # Number of trees
    'max_depth': [3, 5, 7],                   # Maximum depth of a tree
    'learning_rate': [0.01, 0.1, 0.2],        # Step size shrinkage
    'subsample': [0.7, 0.8, 1.0],             # Fraction of samples used per tree
    'colsample_bytree': [0.7, 0.8, 1.0],      # Fraction of features used per tree
    'gamma': [0, 0.1, 0.3],                   # Minimum loss reduction
    'min_child_weight': [1, 3, 5]             # Minimum sum of instance weight for a child
}


# Hyperparameter Tuning using GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)


# Get the Best Hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)


# Train the Final Model with Best Parameters
best_model = grid_search.best_estimator_

# Now you can predict:
y_pred = best_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy with Best Parameters: {accuracy * 100:.2f}%")

# Predict BTTS for a Future Match
team1 = 'Chelsea'
team2 = 'Arsenal'

# Filter the matches that these two teams played with each other in the past
past_matches = data[((data['Team'] == team1) & (data['Opponent'] == team2)) |
                    ((data['Team'] == team2) & (data['Opponent'] == team1))]


# If there is no data, let's give an error message
if past_matches.empty:
    print(f"The past match between {team1} and {team2} has not been found in the past.")
else:


    # Now calculate the mean, ignoring NaN values
    avg_xG = past_matches['xG'].mean()
    avg_xGA = past_matches['xGA'].mean()
    avg_Poss = past_matches['Poss'].mean()
    avg_Sh = past_matches['Sh'].mean()
    avg_SoT = past_matches['SoT'].mean()
    avg_Dist = past_matches['Dist'].mean()
    avg_FK = past_matches['FK'].mean()
    avg_PK = past_matches['PK'].mean()


    new_match = pd.DataFrame({
        'xG': [avg_xG],
        'xGA': [avg_xGA],
        'Poss': [avg_Poss],
        'Sh': [avg_Sh],
        'SoT': [avg_SoT],
        'Dist': [avg_Dist],
        'FK': [avg_FK],
        'PK': [avg_PK],
    })


past_matches


# Before calling roc_auc_score, predict probabilities:
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
# Assuming binary classification, take probability of the positive class

auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {auc}")

    pred_proba = best_model.predict_proba(new_match)

    btts_proba = pred_proba[0][1] * 100 
    no_btts_proba = pred_proba[0][0] * 100 

    if btts_proba > no_btts_proba:
        print(f"Both teams are likely to score in the {team1} vs {team2} match.")
    else:
        print(f"Both teams are unlikely to score in the {team1} vs {team2} match.")

    # Olasılıkları yazdır
    print(f"Possibility of BTTS: %{btts_proba:.2f}")
    print(f"Possibility of No BTTS: %{no_btts_proba:.2f}")