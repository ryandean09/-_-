import numpy as np
from joblib import load


def model_pred(user_full_data):
    # 步驟 1: 加載模型
    model = load(r"./model/XGBoost model_best.joblib")

    # 步驟 2: 準備數據
    # 模型預期的是一個特徵矩陣 X
    user_info = np.array(
        [
            [
                3,
                user_full_data["school_counts"],
                user_full_data["drink_counts"],
                user_full_data["train_counts"],
                user_full_data["youbike_counts"],
                user_full_data["bus_counts"],
                user_full_data["park_counts"],
                user_full_data["night_market_counts"],
                user_full_data["sports_facilities_counts"],
                user_full_data["mrt_counts"],
                user_full_data["movie_theater_counts"],
                user_full_data["hospital_counts"],
                user_full_data["salary_income_median"],
                user_full_data["people_flow_mean"],
                user_full_data["knock_down_price_mean"],
                2,
                user_full_data["road_area_ratio"],
                user_full_data["age"],
                50,
            ]
        ]
    )
    # 注意：這裡的數據應該與模型訓練時的格式相匹配

    # 步驟 3: 使用模型進行預測
    pred_point = model.predict(user_info)

    return {"pred_point": pred_point[0]}
