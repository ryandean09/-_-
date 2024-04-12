'''
這裡在執行的時候要移動到 python\機械學習，不能直接按由上角的執行，會失敗

cd .\python\機械學習\
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split,cross_val_score
import warnings 
warnings.filterwarnings("ignore", category=UserWarning)
from joblib import dump

from sklearn.ensemble import AdaBoostClassifier
from scipy.stats import randint

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


ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_model.fit(X_train, y_train)

y_pred_ada_transformed = ada_model.predict(X_test)

param_grid_ada = {
    "n_estimators": [10, 50, 100, 200],
    "learning_rate": [0.01, 0.1, 1, 10],
}
param_dist_ada = {"n_estimators": randint(50, 500), "learning_rate": [0.01, 0.1, 1, 10]}


grid_search_ada = GridSearchCV(
    AdaBoostClassifier(random_state=42), param_grid_ada, refit=True, verbose=0
)
random_search_ada = RandomizedSearchCV(
    AdaBoostClassifier(random_state=42),
    param_distributions=param_dist_ada,
    n_iter=100,
    refit=True,
    verbose=0,
)

grid_search_ada.fit(X_train, y_train)
random_search_ada.fit(X_train, y_train)

best_model_grid_ada = grid_search_ada.best_estimator_
best_model_random_ada = random_search_ada.best_estimator_


print(
    "Best parameters (GridSearchCV) (最佳參數-網格搜索):", grid_search_ada.best_params_
)
print(
    "Best parameters (RandomizedSearchCV) (最佳參數-隨機搜索):",
    random_search_ada.best_params_,
)

y_pred_ada_best_grid = best_model_grid_ada.predict(X_test)
y_pred_ada_best_random = best_model_random_ada.predict(X_test)

accuracy_ada_best_grid = accuracy_score(y_test, y_pred_ada_best_grid)
accuracy_ada_best_random = accuracy_score(y_test, y_pred_ada_best_random)

if accuracy_ada_best_grid > accuracy_ada_best_random:
    AdaBoost_model_best = best_model_grid_ada
    print("GridSearchCV 的模型表現較好。")
else:
    AdaBoost_model_best = best_model_random_ada
    print("RandomizedSearchCV 的模型表現較好。")

cross_val_scores = cross_val_score(AdaBoost_model_best, X, y, cv=5)  

y_pred_best = AdaBoost_model_best.predict(X_test)

y_pred_best_AdaBoost = AdaBoost_model_best.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_best_AdaBoost)
precision = precision_score(y_test, y_pred_best_AdaBoost, average="macro")
recall = recall_score(y_test, y_pred_best_AdaBoost, average="macro")
f1 = f1_score(y_test, y_pred_best_AdaBoost, average="macro")

accuracy = f"{accuracy * 100:.3f}%"
precision = f"{precision * 100:.3f}%"
recall = f"{recall * 100:.3f}%"
f1 = f"{f1 * 100:.3f}%"

print(f"Accuracy (準確率): {accuracy}")
print(f"Precision (精確率) - Macro Average: {precision}")
print(f"Recall (召回率) - Macro Average: {recall}")
print(f"F1 Score (F1分數) - Macro Average: {f1}")