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

from sklearn.naive_bayes import GaussianNB


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



Bayesion_model = GaussianNB()
Bayesion_model.fit(X_train, y_train)
y_pred_bc = Bayesion_model.predict(X_test)

param_grid = {
    "var_smoothing": np.logspace(0, -15, num=100) }

grid_search_bc = GridSearchCV(
    GaussianNB(),
    param_grid,
    cv=5,
    scoring="accuracy",)

grid_search_bc.fit(X_train, y_train)

best_var_smoothing = grid_search_bc.best_params_["var_smoothing"]

Bayesion_model_best = GaussianNB(var_smoothing=best_var_smoothing)

Bayesion_model_best.fit(X_train, y_train)

# # 保存被訓練過的模型
# dump(svc_best, "svc_best.joblib")

y_pred_best_Bayesion = Bayesion_model_best.predict(X_test)

scores = cross_val_score(y_pred_best_Bayesion, X_train, y_train, cv=5)

accuracy = accuracy_score(y_test, y_pred_best_Bayesion)
precision = precision_score(y_test, y_pred_best_Bayesion, average="macro")
recall = recall_score(y_test, y_pred_best_Bayesion, average="macro")
f1 = f1_score(y_test, y_pred_best_Bayesion, average="macro")

accuracy = f"{accuracy * 100:.3f}%"
precision = f"{precision * 100:.3f}%"
recall = f"{recall * 100:.3f}%"
f1 = f"{f1 * 100:.3f}%"

print(f"Accuracy (準確率): {accuracy}")
print(f"Precision (精確率) - Macro Average: {precision}")
print(f"Recall (召回率) - Macro Average: {recall}")
print(f"F1 Score (F1分數) - Macro Average: {f1}")