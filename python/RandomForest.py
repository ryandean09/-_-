'''
這裡在執行的時候要移動到 python\機械學習，不能直接按由上角的執行，會失敗

cd .\python\機械學習\
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split,cross_val_score
import warnings 
warnings.filterwarnings("ignore", category=UserWarning)
from joblib import dump

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

file_path = r"..\..\CSV\飲料店總表0307final01_補上人氣_補值_hg.xlsx"
dataset = pd.read_excel(file_path)

selected_data = dataset[
    [
        "star","school_counts",
        "drink_counts","train_counts",
        "youbike_counts","bus_counts",
        "park_counts","night_market_counts",
        "sports_facilities_counts","mrt_counts",
        "movie_theater_counts","hospital_counts",
        "salary_income_median","people_flow_mean",
        "knock_down_price_mean","weekend_open",
        "road_area_ratio","age",
        "weekday_working_hours_average","popularity",
    ]
]

selected_data = selected_data[selected_data["popularity"] <= 2206.287]
selected_data = selected_data[selected_data["popularity"] != 0]
min_val = selected_data["popularity"].min()
max_val = selected_data["popularity"].max()
bins = np.linspace(min_val, max_val, 6)
selected_data["popularity_category"] = pd.cut(
    selected_data["popularity"],
    bins=bins,
    include_lowest=True,  
    labels=[0, 1, 2, 3, 4]
)

selected_data["age"] = selected_data["age"].round(2)
selected_data["road_area_ratio"] = selected_data["road_area_ratio"].round(3)

y = selected_data["popularity_category"]
y = y.to_frame()
X = selected_data.drop(["popularity", "popularity_category"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

    
randomforest_model = RandomForestClassifier(random_state=42)
randomforest_model.fit(X_train, y_train)

y_pred_rf = randomforest_model.predict(X_test)

param_grid = {
    "n_estimators": [100, 200, 400, 600],
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [2, 4, 8, 16],
    "max_features": ["auto", "sqrt", "log2"],}

param_dist = {
    "n_estimators": sp_randint(100, 600),
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": sp_randint(2, 11),
    "min_samples_leaf": sp_randint(1, 17),
    "max_features": ["auto", "sqrt", "log2"],}

grid_search = GridSearchCV(randomforest_model, param_grid, cv=5, scoring="accuracy")

random_search = RandomizedSearchCV(
    randomforest_model, param_distributions=param_dist, n_iter=100, cv=5, scoring="accuracy")

grid_search.fit(X_train, y_train)
random_search.fit(X_train, y_train)

if grid_search.best_score_ > random_search.best_score_:
    print("總體最佳參數來自 GridSearchCV:", grid_search.best_params_)
    best_params = grid_search.best_params_
else:
    print("總體最佳參數來自 RandomizedSearchCV:", random_search.best_params_)
    best_params = random_search.best_params_

best_randomforest_model = RandomForestClassifier(**best_params, random_state=42)

best_randomforest_model.fit(X_train, y_train)

y_pred_best = best_randomforest_model.predict(X_test)

# # 保存被訓練過的模型
# dump(best_randomforest_model, "best_randomforest_model.joblib")

y_pred_best_randomforest = best_randomforest_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_best_randomforest)
precision = precision_score(y_test, y_pred_best_randomforest, average="macro")
recall = recall_score(y_test, y_pred_best_randomforest, average="macro")
f1 = f1_score(y_test, y_pred_best_randomforest, average="macro")

accuracy = f"{accuracy * 100:.3f}%"
precision = f"{precision * 100:.3f}%"
recall = f"{recall * 100:.3f}%"
f1 = f"{f1 * 100:.3f}%"

print(f"Accuracy (準確率): {accuracy}")
print(f"Precision (精確率) - Macro Average: {precision}")
print(f"Recall (召回率) - Macro Average: {recall}")
print(f"F1 Score (F1分數) - Macro Average: {f1}")