'''
這裡在執行的時候要移動到 python\機械學習，不能直接按由上角的執行，會失敗

cd .\python\機械學習\
'''

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

def evaluate_model_multi_class(y_test, y_pred):
    '''
    回傳Accuracy (準確率)、Precision (精確率)、Recall (召回率)、F1 Score (F1分數)
    '''
    num_classes = len(np.unique(y_test))
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Accuracy (準確率): {accuracy:.2%}")
    print(f"Precision (精確率): {precision:.2%}")
    print(f"Recall (召回率)e: {recall:.2%}")
    print(f"F1 Score (F1分數): {f1:.2%}")



import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split,cross_val_score
import warnings 
warnings.filterwarnings("ignore", category=UserWarning)
from joblib import dump


import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import GradientBoostingClassifier

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

# C:\Users\D\Desktop\藤豪版\python\Ensemble learning.py
# C:\Users\D\Desktop\藤豪版\python\機械學習\專題模型\模型\AdaBoost_model_best.joblib
model_paths = [
    # "模型\\AdaBoost_model_best.joblib",
    # "模型\\Bayesion_classifier_model_best.joblib",
    "模型\\best_randomforest_model.joblib",
    # "模型\\lr_model_best.joblib",
    # "模型\\MultinomialNB_model_best.joblib",
    # "模型\\svc_best.joblib",
    "模型\\XGBoost_model_best.joblib",
    # "模型\\best_mlp_model.joblib",
]

models = [joblib.load(model_path) for model_path in model_paths]

'''bagging'''
bagging_model = BaggingClassifier(n_estimators=len(models), random_state=42)

X_train_preds = np.array([model.predict(X_train) for model in models]).T
X_test_preds = np.array([model.predict(X_test) for model in models]).T
bagging_model.fit(X_train_preds, y_train)

y_pred_bag = bagging_model.predict(X_test_preds)
acc_bag = accuracy_score(y_test, y_pred_bag)
print(f"Bagging: ")
evaluate_model_multi_class(y_test, y_pred_bag)


'''voting'''
estimators = [(f"model_{i}", model) for i, model in enumerate(models)]

voting_clf = VotingClassifier(estimators=estimators, voting="hard")

voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)

print(f"Voting: ")
evaluate_model_multi_class(y_test, y_pred_voting)

'''stacking'''
estimators = [(f"model_{i}", model) for i, model in enumerate(models)]

stacking_clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression())

stacking_clf.fit(X_train, y_train)

y_pred_stacking = stacking_clf.predict(X_test)

print(f"Stacking: ")
evaluate_model_multi_class(y_test, y_pred_stacking)

'''boosting'''
gb_model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

gb_model.estimators_ = models
gb_model.fit(X_train, y_train)
y_pred_boosting = gb_model.predict(X_test)
print(f"Boosting: ")
evaluate_model_multi_class(y_test, y_pred_boosting)



# # 保存被訓練過的模型
# # dump(svc_best, "svc_best.joblib")