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


from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

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



le = LabelEncoder()
y_train = le.fit_transform(y_train)


xgb_model = xgb.XGBClassifier(
    enable_categorical=True,
)

xgb_model.fit(X_train, y_train)

param_grid = {
    "max_depth": [3, 4, 5],  # 最大深度
    "learning_rate": [0.1, 0.01, 0.001],  # 學習率
    "n_estimators": [100, 200, 300],  # 樹的數量
    "objective": ["multi:softmax", "multi:softprob"],  # 目標函數
    "subsample": [0.6, 0.8, 1],  # 子樣本比例
    "colsample_bytree": [0.8, 1, 1.2],  # 每棵樹隨機選擇特徵的比例
}



grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="precision_macro",
    n_jobs=-1,
    cv=3,
    verbose=2,
)


grid_search.fit(X_train, y_train)

xgb_model_best = grid_search.best_estimator_

y_pred_best_xgb = xgb_model_best.predict(X_test)

accuracy_best = accuracy_score(y_test, y_pred_best_xgb)
classification_report_best = classification_report(y_test, y_pred_best_xgb)



y_pred_best_lr = xgb_model_best.predict(X_test)

scores = cross_val_score(xgb_model_best, X_train, y_train, cv=5)

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


'''顯示特徵重要性'''
import pandas as pd
import matplotlib.pyplot as plt

feature_names = X_train.columns
importances = xgb_model_best.feature_importances_
feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": importances})


feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False 


feature_name_map = {
    "knock_down_price_mean": "地區租屋平均值",
    "road_area_ratio": "道路面積比率",
    "mrt_counts": "捷運站數量",
    "weekend_open": "週末開放",
    "star": "星級",
    "sports_facilities_counts": "體育設施數量",
    "people_flow_mean": "人流量平均值",
    "salary_income_median": "薪水收入中位數",
    "youbike_counts": "YouBike站點數量",
    "weekday_working_hours_average": "工作日平均工時",
    "school_counts":"學校數量",
    "drink_counts":"飲料店數",
    "train_counts":"火車站數量",
    "bus_counts":"公車站數量",
    "park_counts":"公園數量",
    "night_market_counts":"夜市數量",
    "movie_theater_counts":"電影數量",
    "hospital_counts":"醫院數量",
    "age":"年齡"
}


feature_importance["Feature"] = feature_importance["Feature"].map(feature_name_map)


plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance["Feature"][:10],  
    feature_importance["Importance"][:10],
)
plt.xlabel("重要性(Importance)")
plt.ylabel("特徵(Feature)")
plt.title("前 10 特徵重要性 - XGBoost")
plt.gca().invert_yaxis()  
plt.show()