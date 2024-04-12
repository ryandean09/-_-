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

from sklearn.linear_model import LogisticRegression

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


lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
classification_report_lr = classification_report(y_test, y_pred_lr)

C_values = np.logspace(-3, 3, 6)

original_param_grid = {"C": C_values, "penalty": ["l1", "l2"]}
original_grid_search = GridSearchCV(
    LogisticRegression(
        max_iter=1000, random_state=42, solver="liblinear"
    ), 
    original_param_grid,
    cv=3,
    scoring="accuracy",
)

original_grid_search.fit(X_train, y_train)
original_best_params = original_grid_search.best_params_
original_best_score = original_grid_search.best_score_

elastic_param_grid = {
    "C": C_values,
    "l1_ratio": np.linspace(0, 1, 5),
}
elastic_grid_search = GridSearchCV(
    LogisticRegression(
        max_iter=1000, random_state=42, penalty="elasticnet", solver="saga"
    ),
    elastic_param_grid,
    cv=3,
    scoring="accuracy",
)

elastic_grid_search.fit(X_train, y_train)
elastic_best_params = elastic_grid_search.best_params_
elastic_best_score = elastic_grid_search.best_score_

if original_best_score > elastic_best_score:
    print("原始模型較好。")
    lr_model_best = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="liblinear",
        **original_best_params 
    )
else:
    print("Elastic Net 模型較好。")
    lr_model_best = LogisticRegression(
        max_iter=1000,
        random_state=42,
        penalty="elasticnet",
        solver="saga",
        **elastic_best_params 
    )

lr_model_best.fit(X_train, y_train)

# # 保存被訓練過的模型
# dump(svc_best, "svc_best.joblib")

y_pred_best_lr = lr_model_best.predict(X_test)

scores = cross_val_score(lr_model_best, X_train, y_train, cv=5)

accuracy = accuracy_score(y_test, y_pred_best_lr)
precision = precision_score(y_test, y_pred_best_lr, average="macro")
recall = recall_score(y_test, y_pred_best_lr, average="macro")
f1 = f1_score(y_test, y_pred_best_lr, average="macro")

accuracy = f"{accuracy * 100:.3f}%"
precision = f"{precision * 100:.3f}%"
recall = f"{recall * 100:.3f}%"
f1 = f"{f1 * 100:.3f}%"

print(f"Accuracy (準確率): {accuracy}")
print(f"Precision (精確率) - Macro Average: {precision}")
print(f"Recall (召回率) - Macro Average: {recall}")
print(f"F1 Score (F1分數) - Macro Average: {f1}")