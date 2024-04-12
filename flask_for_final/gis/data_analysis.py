import os
import re

import geocoder
import geopandas as gpd
import pandas as pd
from geopandas.tools import sjoin
from geopy.geocoders import Nominatim
from shapely.geometry import Point


# 環域分析
def buffer_analysis(address):
    # 輸入地址轉經緯度
    coordinates = geocoder.arcgis(address).latlng
    if coordinates:
        # 給定經度、緯度
        longitude = coordinates[1]  # 經度
        latitude = coordinates[0]  # 緯度

        # 創建中心點
        center_point = Point(longitude, latitude)
        # 建立中心點gdf
        center_gdf = gpd.GeoDataFrame(geometry=[center_point], crs="epsg:4326")
        # 設置原始數據的 CRS 為 WGS 84

        # 載入資料集
        # 指定存放CSV文件的資料夾路徑

        base_folder = os.path.abspath(os.path.dirname(__file__))

        # 雙北飲料店
        drink_df = gpd.read_file(f"{base_folder}\\Dataset\\drink.csv")
        drink_gdf = gpd.GeoDataFrame(
            drink_df,
            geometry=[Point(xy) for xy in zip(drink_df.longitude, drink_df.latitude)],
            crs="epsg:4326",
        )

        # 雙北火車
        train_df = gpd.read_file(f"{base_folder}\\Dataset\\train.csv")
        train_gdf = gpd.GeoDataFrame(
            train_df,
            geometry=[Point(xy) for xy in zip(train_df.longitude, train_df.latitude)],
            crs="epsg:4326",
        )

        # 雙北公車站
        bus_df = gpd.read_file(f"{base_folder}\\Dataset\\bus.csv")
        bus_gdf = gpd.GeoDataFrame(
            bus_df,
            geometry=[Point(xy) for xy in zip(bus_df.longitude, bus_df.latitude)],
            crs="epsg:4326",
        )

        # 雙北youbike站
        youbike_df = gpd.read_file(f"{base_folder}\\Dataset\\youbike.csv")
        youbike_gdf = gpd.GeoDataFrame(
            youbike_df,
            geometry=[
                Point(xy) for xy in zip(youbike_df.longitude, youbike_df.latitude)
            ],
            crs="epsg:4326",
        )

        # 雙北公園
        park_df = gpd.read_file(f"{base_folder}\\Dataset\\park.csv")
        park_gdf = gpd.GeoDataFrame(
            park_df,
            geometry=[Point(xy) for xy in zip(park_df.longitude, park_df.latitude)],
            crs="epsg:4326",
        )

        # 雙北夜市
        night_market_df = gpd.read_file(f"{base_folder}\\Dataset\\night_market.csv")
        night_market_gdf = gpd.GeoDataFrame(
            night_market_df,
            geometry=[
                Point(xy)
                for xy in zip(night_market_df.longitude, night_market_df.latitude)
            ],
            crs="epsg:4326",
        )

        # 雙北捷運
        mrt_df = gpd.read_file(f"{base_folder}\\Dataset\\mrt.csv")
        mrt_gdf = gpd.GeoDataFrame(
            mrt_df,
            geometry=[Point(xy) for xy in zip(mrt_df.longitude, mrt_df.latitude)],
            crs="epsg:4326",
        )

        # 雙北學校
        sports_facilities_df = gpd.read_file(
            f"{base_folder}\\Dataset\\sports_facilities.csv"
        )
        sports_facilities_gdf = gpd.GeoDataFrame(
            sports_facilities_df,
            geometry=[
                Point(xy)
                for xy in zip(
                    sports_facilities_df.longitude, sports_facilities_df.latitude
                )
            ],
            crs="epsg:4326",
        )

        # 雙北電影院
        movie_theater_df = gpd.read_file(f"{base_folder}\\Dataset\\movie_theater.csv")
        movie_theater_gdf = gpd.GeoDataFrame(
            movie_theater_df,
            geometry=[
                Point(xy)
                for xy in zip(movie_theater_df.longitude, movie_theater_df.latitude)
            ],
            crs="epsg:4326",
        )

        # 雙北學校
        school_df = gpd.read_file(f"{base_folder}\\Dataset\\school.csv")
        school_gdf = gpd.GeoDataFrame(
            school_df,
            geometry=[Point(xy) for xy in zip(school_df.longitude, school_df.latitude)],
            crs="epsg:4326",
        )

        # 雙北醫院
        hospital_df = gpd.read_file(f"{base_folder}\\Dataset\\hospital.csv")
        hospital_gdf = gpd.GeoDataFrame(
            hospital_df,
            geometry=[
                Point(xy) for xy in zip(hospital_df.longitude, hospital_df.latitude)
            ],
            crs="epsg:4326",
        )

        # 轉換坐標系統到 Web Mercator (epsg:3826(TWD97 121分帶)) 以計算以公尺為單位距離
        center_gdf = center_gdf.to_crs(epsg=3826)
        school_gdf = school_gdf.to_crs(epsg=3826)
        drink_gdf = drink_gdf.to_crs(epsg=3826)
        train_gdf = train_gdf.to_crs(epsg=3826)
        bus_gdf = bus_gdf.to_crs(epsg=3826)
        youbike_gdf = youbike_gdf.to_crs(epsg=3826)
        park_gdf = park_gdf.to_crs(epsg=3826)
        night_market_gdf = night_market_gdf.to_crs(epsg=3826)
        sports_facilities_gdf = sports_facilities_gdf.to_crs(epsg=3826)
        mrt_gdf = mrt_gdf.to_crs(epsg=3826)
        movie_theater_gdf = movie_theater_gdf.to_crs(epsg=3826)
        hospital_gdf = hospital_gdf.to_crs(epsg=3826)

        # 建立中心點的id(唯一值)
        center_gdf["shop_id"] = range(len(center_gdf))

        # 建立緩衝區(輪廓) GeoDataFrame
        buffer_gdf = gpd.GeoDataFrame(center_gdf[["shop_id", "geometry"]].copy())
        buffer_gdf["geometry"] = buffer_gdf.geometry.buffer(
            1000, resolution=99
        )  # 建立1公里緩衝區(輪廓)

        # 空間連接(join)
        school_joined_gdf = sjoin(
            buffer_gdf, school_gdf, how="inner", predicate="contains"
        )
        drink_joined_gdf = sjoin(
            buffer_gdf, drink_gdf, how="inner", predicate="contains"
        )
        train_joined_gdf = sjoin(
            buffer_gdf, train_gdf, how="inner", predicate="contains"
        )
        bus_joined_gdf = sjoin(buffer_gdf, bus_gdf, how="inner", predicate="contains")
        youbike_joined_gdf = sjoin(
            buffer_gdf, youbike_gdf, how="inner", predicate="contains"
        )
        park_joined_gdf = sjoin(buffer_gdf, park_gdf, how="inner", predicate="contains")
        night_market_joined_gdf = sjoin(
            buffer_gdf, night_market_gdf, how="inner", predicate="contains"
        )
        sports_facilities_joined_gdf = sjoin(
            buffer_gdf, sports_facilities_gdf, how="inner", predicate="contains"
        )
        mrt_joined_gdf = sjoin(buffer_gdf, mrt_gdf, how="inner", predicate="contains")
        movie_theater_joined_gdf = sjoin(
            buffer_gdf, movie_theater_gdf, how="inner", predicate="contains"
        )
        hospital_joined_gdf = sjoin(
            buffer_gdf, hospital_gdf, how="inner", predicate="contains"
        )

        # 執行統計每個緩沖區內的設施數量
        school_counts = school_joined_gdf.groupby("shop_id").size()
        drink_counts = drink_joined_gdf.groupby("shop_id").size()
        train_counts = train_joined_gdf.groupby("shop_id").size()
        bus_counts = bus_joined_gdf.groupby("shop_id").size()
        youbike_counts = youbike_joined_gdf.groupby("shop_id").size()
        park_counts = park_joined_gdf.groupby("shop_id").size()
        night_market_counts = night_market_joined_gdf.groupby("shop_id").size()
        sports_facilities_counts = sports_facilities_joined_gdf.groupby(
            "shop_id"
        ).size()
        mrt_counts = mrt_joined_gdf.groupby("shop_id").size()
        movie_theater_counts = movie_theater_joined_gdf.groupby("shop_id").size()
        hospital_counts = hospital_joined_gdf.groupby("shop_id").size()

        # 需要將計數結果與原始的多邊形 GeoDataFrame 進行合併
        # 為了確保即使是數值為 0 的多邊形也能被統計
        buffer_gdf["school_counts"] = buffer_gdf.index.map(school_counts).fillna(0)
        buffer_gdf["drink_counts"] = buffer_gdf.index.map(drink_counts).fillna(0)
        buffer_gdf["train_counts"] = buffer_gdf.index.map(train_counts).fillna(0)
        buffer_gdf["bus_counts"] = buffer_gdf.index.map(bus_counts).fillna(0)
        buffer_gdf["youbike_counts"] = buffer_gdf.index.map(youbike_counts).fillna(0)
        buffer_gdf["park_counts"] = buffer_gdf.index.map(park_counts).fillna(0)
        buffer_gdf["night_market_counts"] = buffer_gdf.index.map(
            night_market_counts
        ).fillna(0)
        buffer_gdf["sports_facilities_counts"] = buffer_gdf.index.map(
            sports_facilities_counts
        ).fillna(0)
        buffer_gdf["mrt_counts"] = buffer_gdf.index.map(mrt_counts).fillna(0)
        buffer_gdf["movie_theater_counts"] = buffer_gdf.index.map(
            movie_theater_counts
        ).fillna(0)
        buffer_gdf["hospital_counts"] = buffer_gdf.index.map(hospital_counts).fillna(0)

        # 設施數量
        school_counts = int(buffer_gdf["school_counts"].values[0])
        drink_counts = int(buffer_gdf["drink_counts"].values[0])
        train_counts = int(buffer_gdf["train_counts"].values[0])
        bus_counts = int(buffer_gdf["bus_counts"].values[0])
        youbike_counts = int(buffer_gdf["youbike_counts"].values[0])
        park_counts = int(buffer_gdf["park_counts"].values[0])
        night_market_counts = int(buffer_gdf["night_market_counts"].values[0])
        sports_facilities_counts = int(buffer_gdf["sports_facilities_counts"].values[0])
        mrt_counts = int(buffer_gdf["mrt_counts"].values[0])
        movie_theater_counts = int(buffer_gdf["movie_theater_counts"].values[0])
        hospital_counts = int(buffer_gdf["hospital_counts"].values[0])
        return {
            "address": address,
            "latitude": latitude,
            "longitude": longitude,
            "school_counts": school_counts,
            "drink_counts": drink_counts,
            "train_counts": train_counts,
            "bus_counts": bus_counts,
            "youbike_counts": youbike_counts,
            "park_counts": park_counts,
            "night_market_counts": night_market_counts,
            "sports_facilities_counts": sports_facilities_counts,
            "mrt_counts": mrt_counts,
            "movie_theater_counts": movie_theater_counts,
            "hospital_counts": hospital_counts,
        }

    else:
        return {"error": "Invalid address"}


# 取得行政區、鄰里
def user_district(address_info):

    latitude = address_info["latitude"]
    longitude = address_info["longitude"]

    # 經緯度座標轉地址
    def get_address_from_coordinates(latitude, longitude):
        geolocator = Nominatim(
            user_agent="your_app_name"
        )  # 設定你的應用程式名稱作為 user_agent
        location = geolocator.reverse(
            (latitude, longitude), language="zh-tw"
        )  # 設定查詢語言為繁體中文
        address = location.address if location else "找不到地址"
        return address

    address = get_address_from_coordinates(latitude, longitude)
    re_addess = r"(.{2}里),.(.{2}區),"
    address_search = re.search(re_addess, address)

    district = address_search[2]
    neighborhood = address_search[1]

    add_district = {"district": district, "neighborhood": neighborhood}

    address_info.update(add_district)

    return address_info


def user_data(address_district_info):
    base_folder = os.path.abspath(os.path.dirname(__file__))

    # 地區薪資中位數
    salary_df = pd.read_csv(f"{base_folder}\\Dataset\\area\\salary.csv")
    salary_district_mask = salary_df[
        salary_df["district"] == address_district_info["district"]
    ]
    salary_neighborhood_mask = salary_district_mask[
        salary_district_mask["neighborhood"] == address_district_info["neighborhood"]
    ]
    salary_income_median = salary_neighborhood_mask["median"].iloc[0]

    # 地區平均年齡
    age_df = pd.read_csv(f"{base_folder}\\Dataset\\area\\age.csv")
    age_area_mask = age_df["district"] == address_district_info["district"]
    age = age_df.loc[age_area_mask, "age"].iloc[0]

    # 地區人流
    people_flow_mean_df = pd.read_csv(
        f"{base_folder}\\Dataset\\area\\people_flow_mean.csv"
    )
    people_flow_mean_area_mask = (
        people_flow_mean_df["district"] == address_district_info["district"]
    )
    people_flow_mean = people_flow_mean_df.loc[
        people_flow_mean_area_mask, "people_flow_mean"
    ].iloc[0]

    # 道路面積比例
    road_area_df = pd.read_csv(f"{base_folder}\\Dataset\\area\\Road_area_ratio.csv")
    road_area_mask = road_area_df["district"] == address_district_info["district"]
    road_area_ratio = road_area_df.loc[road_area_mask, "Road_area_ratio"].iloc[0]

    # 地區單坪成交租金
    knock_down_price_df = pd.read_csv(
        f"{base_folder}\\Dataset\\area\\knock_down_price_mean.csv"
    )
    knock_down_price_mask = (
        knock_down_price_df["district"] == address_district_info["district"]
    )
    knock_down_price_mean = knock_down_price_df.loc[
        knock_down_price_mask, "knock_down_price_mean"
    ].iloc[0]

    add_scalar_data = {
        "salary_income_median": salary_income_median,
        "age": age,
        "people_flow_mean": people_flow_mean,
        "road_area_ratio": road_area_ratio,
        "knock_down_price_mean": knock_down_price_mean,
    }

    address_district_info.update(add_scalar_data)

    return address_district_info
