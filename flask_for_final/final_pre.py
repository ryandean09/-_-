import os
import re

import geocoder
import geopandas as gpd
import gis.data_analysis as gis
import model.model_counting as usemodel
import requests
from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from geopandas.tools import sjoin
from geopy.geocoders import Nominatim
from shapely.geometry import Point
from sqlalchemy import desc

# 資料庫設定
db_settings = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "P%40ssw0rd",
    "db": "final_pro",
    "charset": "utf8",
}


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"mysql://{db_settings['user']}:{db_settings['password']}@{db_settings['host']}:{db_settings['port']}/{db_settings['db']}?charset={db_settings['charset']}"
)
db = SQLAlchemy(app)


class LocationData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    address = db.Column(db.String(255))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    school_counts = db.Column(db.Integer)
    drink_counts = db.Column(db.Integer)
    train_counts = db.Column(db.Integer)
    bus_counts = db.Column(db.Integer)
    park_counts = db.Column(db.Integer)
    night_market_counts = db.Column(db.Integer)
    sports_facilities_counts = db.Column(db.Integer)
    mrt_counts = db.Column(db.Integer)
    movie_theater_counts = db.Column(db.Integer)
    hospital_counts = db.Column(db.Integer)


# 首頁顯示
@app.route("/")
def index():

    return render_template("index.html")


# 在你的 Flask 應用程式中新增這個路由
@app.route("/page2.html")
def page2():
    # 從資料庫中提取最後十筆資料，以最後更新時間排序
    data_from_db = LocationData.query.order_by(desc(LocationData.id)).limit(10).all()

    # 從資料中提取地址
    search_history = [record.address for record in data_from_db]
    return render_template("page2.html", search_history=search_history)


# 在你的 Flask 應用程式中新增這個路由
@app.route("/bi.html")
def bi():
    return render_template("bi.html")


# 使用者輸入地址 轉經緯度去找變數(完成)
@app.route("/get_coordinates", methods=["GET", "POST"])
def get_coordinates():
    data = request.json
    address = data[
        "address"
    ]  ########################## 使用者輸入的地址###########################
    # 輸入地址轉經緯度
    coordinates = geocoder.arcgis(address).latlng
    if coordinates:
        # 給定經度、緯度
        longitude = coordinates[1]  #####################leftlet經緯度################
        latitude = coordinates[0]  #####################0是緯度######################

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
        park_joined_gdf = sjoin(buffer_gdf, park_gdf, how="inner", predicate="contains")
        night_market_joined_gdf = sjoin(
            buffer_gdf, night_market_gdf, how="inner", predicate="contains"
        )
        youbike_joined_gdf = sjoin(
            buffer_gdf, youbike_gdf, how="inner", predicate="contains"
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
        # 執行統計每個緩沖區內的設施數量

        buffer_gdf["drink_counts"] = buffer_gdf.index.map(drink_counts).fillna(0)
        buffer_gdf["school_counts"] = buffer_gdf.index.map(drink_counts).fillna(0)
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

        ##########變數 個數量#########
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

        print(
            f"學校數量 : {school_counts}"
            "\n"
            f"飲料店數量 : {drink_counts}"
            "\n"
            f"火車站數量 : {train_counts}"
            "\n"
            f"公車站數量 : {bus_counts}"
            "\n"
            f"公園數量 : {park_counts}"
            "\n"
            f"夜市數量 : {night_market_counts}"
            "\n"
            f"運動設施數量 : {sports_facilities_counts}"
            "\n"
            f"捷運站數量 : {mrt_counts}"
            "\n"
            f"電影院數量 : {movie_theater_counts}"
            "\n"
            f"醫院數量 : {hospital_counts}"
            "\n"
        )

        # Create a new LocationData instance and save it to the database
        location_data = LocationData(
            address=address,
            latitude=latitude,
            longitude=longitude,
            school_counts=school_counts,
            drink_counts=drink_counts,
            train_counts=train_counts,
            bus_counts=bus_counts,
            park_counts=park_counts,
            night_market_counts=night_market_counts,
            sports_facilities_counts=sports_facilities_counts,
            mrt_counts=mrt_counts,
            movie_theater_counts=movie_theater_counts,
            hospital_counts=hospital_counts,
        )
        db.session.add(location_data)
        db.session.commit()

        return jsonify(
            {
                "address": address,
                "latitude": latitude,
                "longitude": longitude,
                "學校數量": school_counts,
                "飲料店數量": drink_counts,
                "火車站數量": train_counts,
                "公車站數量": bus_counts,
                "youbike_counts": youbike_counts,
                "公園數量": park_counts,
                "夜市數量": night_market_counts,
                "運動設施數量": sports_facilities_counts,
                "捷運站數量": mrt_counts,
                "電影院數量": movie_theater_counts,
                "醫院數量": hospital_counts,
                "行政區": district,
                "鄰里": neighborhood,
            }
        )
    else:
        return jsonify({"error": "Invalid address"})


# 接10大飲料店  (等csv)
@app.route("/top10_brand", methods=["POST"])
def top10_brand():
    data = request.json
    address = data[
        "address"
    ]  ########################## 使用者輸入的地址###########################
    # 輸入地址轉經緯度
    coordinates = geocoder.arcgis(address).latlng
    if coordinates:
        # 給定經度、緯度
        longitude = coordinates[1]  #####################leftlet經緯度################
        latitude = coordinates[0]  #####################0是緯度######################

        # 創建中心點
        center_point = Point(longitude, latitude)
        # 建立中心點gdf
        center_gdf = gpd.GeoDataFrame(geometry=[center_point], crs="epsg:4326")
        # 設置原始數據的 CRS 為 WGS 84

    base_folder = os.path.abspath(os.path.dirname(__file__))

    COCO_df = gpd.read_file(f"{base_folder}\\Dataset\\top10\\COCO.csv")
    COCO_gdf = gpd.GeoDataFrame(
        COCO_df,
        geometry=[Point(xy) for xy in zip(COCO_df.longitude, COCO_df.latitude)],
        crs="epsg:4326",
    )

    五桐號_df = gpd.read_file(f"{base_folder}\\Dataset\\top10\\五桐號.csv")
    五桐號_gdf = gpd.GeoDataFrame(
        五桐號_df,
        geometry=[Point(xy) for xy in zip(五桐號_df.longitude, 五桐號_df.latitude)],
        crs="epsg:4326",
    )

    五十嵐_df = gpd.read_file(f"{base_folder}\\Dataset\\top10\\五十嵐.csv")
    五十嵐_gdf = gpd.GeoDataFrame(
        五十嵐_df,
        geometry=[Point(xy) for xy in zip(五十嵐_df.longitude, 五十嵐_df.latitude)],
        crs="epsg:4326",
    )

    大苑子_df = gpd.read_file(f"{base_folder}\\Dataset\\top10\\大苑子.csv")
    大苑子_gdf = gpd.GeoDataFrame(
        大苑子_df,
        geometry=[Point(xy) for xy in zip(大苑子_df.longitude, 大苑子_df.latitude)],
        crs="epsg:4326",
    )

    先喝道_df = gpd.read_file(f"{base_folder}\\Dataset\\top10\\先喝道.csv")
    先喝道_gdf = gpd.GeoDataFrame(
        先喝道_df,
        geometry=[Point(xy) for xy in zip(先喝道_df.longitude, 先喝道_df.latitude)],
        crs="epsg:4326",
    )

    麻古茶坊_df = gpd.read_file(f"{base_folder}\\Dataset\\top10\\麻古茶坊.csv")
    麻古茶坊_gdf = gpd.GeoDataFrame(
        麻古茶坊_df,
        geometry=[Point(xy) for xy in zip(麻古茶坊_df.longitude, 麻古茶坊_df.latitude)],
        crs="epsg:4326",
    )

    一沐日_df = gpd.read_file(f"{base_folder}\\Dataset\\top10\\一沐日.csv")
    一沐日_gdf = gpd.GeoDataFrame(
        一沐日_df,
        geometry=[Point(xy) for xy in zip(一沐日_df.longitude, 一沐日_df.latitude)],
        crs="epsg:4326",
    )

    春水堂_df = gpd.read_file(f"{base_folder}\\Dataset\\top10\\春水堂.csv")
    春水堂_gdf = gpd.GeoDataFrame(
        春水堂_df,
        geometry=[Point(xy) for xy in zip(春水堂_df.longitude, 春水堂_df.latitude)],
        crs="epsg:4326",
    )

    清心福全_df = gpd.read_file(f"{base_folder}\\Dataset\\top10\\清心福全.csv")
    清心福全_gdf = gpd.GeoDataFrame(
        清心福全_df,
        geometry=[Point(xy) for xy in zip(清心福全_df.longitude, 清心福全_df.latitude)],
        crs="epsg:4326",
    )

    珍煮丹_df = gpd.read_file(f"{base_folder}\\Dataset\\top10\\珍煮丹.csv")
    珍煮丹_gdf = gpd.GeoDataFrame(
        珍煮丹_df,
        geometry=[Point(xy) for xy in zip(珍煮丹_df.longitude, 珍煮丹_df.latitude)],
        crs="epsg:4326",
    )

    center_gdf = center_gdf.to_crs(epsg=3826)
    COCO_gdf = COCO_gdf.to_crs(epsg=3826)
    五桐號_gdf = 五桐號_gdf.to_crs(epsg=3826)
    五十嵐_gdf = 五十嵐_gdf.to_crs(epsg=3826)
    大苑子_gdf = 大苑子_gdf.to_crs(epsg=3826)
    先喝道_gdf = 先喝道_gdf.to_crs(epsg=3826)
    麻古茶坊_gdf = 麻古茶坊_gdf.to_crs(epsg=3826)
    一沐日_gdf = 一沐日_gdf.to_crs(epsg=3826)
    春水堂_gdf = 春水堂_gdf.to_crs(epsg=3826)
    清心福全_gdf = 清心福全_gdf.to_crs(epsg=3826)
    珍煮丹_gdf = 珍煮丹_gdf.to_crs(epsg=3826)

    # 建立中心點的id(唯一值)
    center_gdf["shop_id"] = range(len(center_gdf))

    # 建立緩衝區(輪廓) GeoDataFrame
    buffer_gdf = gpd.GeoDataFrame(center_gdf[["shop_id", "geometry"]].copy())
    buffer_gdf["geometry"] = buffer_gdf.geometry.buffer(
        1000, resolution=99
    )  # 建立1公里緩衝區(輪廓)

    # 空間連接(join)
    COCO_gdf_joined_gdf = sjoin(
        buffer_gdf, COCO_gdf, how="inner", predicate="contains"
    )  # 空間連接(join)
    五十嵐_gdf_joined_gdf = sjoin(
        buffer_gdf, 五十嵐_gdf, how="inner", predicate="contains"
    )  # 空間連接(join)
    大苑子_gdf_joined_gdf = sjoin(
        buffer_gdf, 大苑子_gdf, how="inner", predicate="contains"
    )  # 空間連接(join)
    先喝道_gdf_joined_gdf = sjoin(
        buffer_gdf, 先喝道_gdf, how="inner", predicate="contains"
    )  # 空間連接(join)
    麻古茶坊_gdf_joined_gdf = sjoin(
        buffer_gdf, 麻古茶坊_gdf, how="inner", predicate="contains"
    )  # 空間連接(join)
    一沐日_gdf_joined_gdf = sjoin(
        buffer_gdf, 一沐日_gdf, how="inner", predicate="contains"
    )  # 空間連接(join)
    春水堂_gdf_joined_gdf = sjoin(
        buffer_gdf, 春水堂_gdf, how="inner", predicate="contains"
    )  # 空間連接(join)
    五桐號_gdf_joined_gdf = sjoin(
        buffer_gdf, 五桐號_gdf, how="inner", predicate="contains"
    )  # 空間連接(join)
    清心福全_gdf_joined_gdf = sjoin(
        buffer_gdf, 清心福全_gdf, how="inner", predicate="contains"
    )  # 空間連接(join)
    珍煮丹_gdf_joined_gdf = sjoin(
        buffer_gdf, 珍煮丹_gdf, how="inner", predicate="contains"
    )

    # 執行統計每個緩沖區內的設施數量
    coco_counts = COCO_gdf_joined_gdf.groupby(
        "shop_id"
    ).size()  # 執行統計每個緩沖區內的設施數量
    五十嵐_counts = 五十嵐_gdf_joined_gdf.groupby(
        "shop_id"
    ).size()  # 執行統計每個緩沖區內的設施數量
    大苑子_counts = 大苑子_gdf_joined_gdf.groupby(
        "shop_id"
    ).size()  # 執行統計每個緩沖區內的設施數量
    先喝道_counts = 先喝道_gdf_joined_gdf.groupby(
        "shop_id"
    ).size()  # 執行統計每個緩沖區內的設施數量
    麻古茶坊_counts = 麻古茶坊_gdf_joined_gdf.groupby(
        "shop_id"
    ).size()  # 執行統計每個緩沖區內的設施數量
    一沐日_counts = 一沐日_gdf_joined_gdf.groupby(
        "shop_id"
    ).size()  # 執行統計每個緩沖區內的設施數量
    春水堂_counts = 春水堂_gdf_joined_gdf.groupby(
        "shop_id"
    ).size()  # 執行統計每個緩沖區內的設施數量
    清心福全_counts = 清心福全_gdf_joined_gdf.groupby(
        "shop_id"
    ).size()  # 執行統計每個緩沖區內的設施數量
    珍煮丹_counts = 珍煮丹_gdf_joined_gdf.groupby(
        "shop_id"
    ).size()  # 執行統計每個緩沖區內的設施數量
    五桐號_counts = 五桐號_gdf_joined_gdf.groupby("shop_id").size()

    buffer_gdf["coco_counts"] = buffer_gdf.index.map(coco_counts).fillna(0)
    buffer_gdf["五十嵐_counts"] = buffer_gdf.index.map(五十嵐_counts).fillna(0)
    buffer_gdf["大苑子_counts"] = buffer_gdf.index.map(大苑子_counts).fillna(0)
    buffer_gdf["先喝道_counts"] = buffer_gdf.index.map(先喝道_counts).fillna(0)
    buffer_gdf["麻古茶坊_counts"] = buffer_gdf.index.map(麻古茶坊_counts).fillna(0)
    buffer_gdf["一沐日_counts"] = buffer_gdf.index.map(一沐日_counts).fillna(0)
    buffer_gdf["春水堂_counts"] = buffer_gdf.index.map(春水堂_counts).fillna(0)
    buffer_gdf["清心福全_counts"] = buffer_gdf.index.map(清心福全_counts).fillna(0)
    buffer_gdf["珍煮丹_counts"] = buffer_gdf.index.map(珍煮丹_counts).fillna(0)
    buffer_gdf["五桐號_counts"] = buffer_gdf.index.map(五桐號_counts).fillna(0)

    coco_counts = int(buffer_gdf["coco_counts"].values[0])
    五十嵐_counts = int(buffer_gdf["五十嵐_counts"].values[0])
    大苑子_counts = int(buffer_gdf["大苑子_counts"].values[0])
    先喝道_counts = int(buffer_gdf["先喝道_counts"].values[0])
    麻古茶坊_counts = int(buffer_gdf["麻古茶坊_counts"].values[0])
    一沐日_counts = int(buffer_gdf["一沐日_counts"].values[0])
    春水堂_counts = int(buffer_gdf["春水堂_counts"].values[0])
    清心福全_counts = int(buffer_gdf["清心福全_counts"].values[0])
    珍煮丹_counts = int(buffer_gdf["珍煮丹_counts"].values[0])
    五桐號_counts = int(buffer_gdf["五桐號_counts"].values[0])

    print(
        {
            "50嵐": 五十嵐_counts,
            "大苑子": 大苑子_counts,
            "珍煮丹": 珍煮丹_counts,
            "先喝道": 先喝道_counts,
            "CoCo都可": coco_counts,
            "麻古茶坊": 麻古茶坊_counts,
            "一沐日": 一沐日_counts,
            "五桐號": 五桐號_counts,
            "春水堂": 春水堂_counts,
            "清心福全": 清心福全_counts,
        }
    )

    return jsonify(
        {
            "五十嵐": 五十嵐_counts,
            "大苑子": 大苑子_counts,
            "珍煮丹": 珍煮丹_counts,
            "先喝道": 先喝道_counts,
            "CoCo都可": coco_counts,
            "麻古茶坊": 麻古茶坊_counts,
            "一沐日": 一沐日_counts,
            "五桐號": 五桐號_counts,
            "春水堂": 春水堂_counts,
            "清心福全": 清心福全_counts,
        }
    )


# 顯示使用者的輸入
@app.route("/save_user_input", methods=["GET", "POST"])
def save_user_input():
    try:
        # 從請求中提取資料
        data = request.json
        address = data["address"]
        latitude = data["latitude"]
        longitude = data["longitude"]

        # 根據您的模型進行其他資料提取

        # 建立新的 LocationData 實例
        new_location_data = LocationData(
            address=address,
            latitude=latitude,
            longitude=longitude,
            # 根據您的模型添加其他欄位
        )

        # 將新實例添加到資料庫會話
        db.session.add(new_location_data)

        # 將更改提交到資料庫
        db.session.commit()

        return jsonify({"message": "使用者輸入成功保存"})
    except Exception as e:
        return jsonify({"error": str(e)})


# 顯示歷史紀錄
@app.route("/search_history")
def search_history():
    # 從資料庫中提取資料，假設您有一個名為LocationData的模型
    data_from_db = LocationData.query.all()

    # 從資料中提取地址
    search_history = [record.address for record in data_from_db]
    print(search_history)

    # 使用搜尋歷史資料渲染模板
    return render_template("page2.html", search_history=search_history)


# 送到model接model資料
@app.route("/load_and_get_model", methods=["POST"])
def load_and_get_model():
    data = request.json
    address = data["address"]
    address_info = gis.buffer_analysis(address)
    address_district_info = gis.user_district(address_info)
    user_full_data = gis.user_data(address_district_info)
    print(user_full_data)

    reccomanding_grade = usemodel.model_pred(user_full_data)
    print(f"Rec: {reccomanding_grade}")

    for key, value in reccomanding_grade.items():
        reccomanding_grade[key] = int(value)

    if reccomanding_grade[key] == 0:
        reccomanding_grade[key] = "極度不推薦"
    elif reccomanding_grade[key] == 1:
        reccomanding_grade[key] = "不推薦"
    elif reccomanding_grade[key] == 2:
        reccomanding_grade[key] = "普通"
    elif reccomanding_grade[key] == 3:
        reccomanding_grade[key] = "推薦"
    else:
        reccomanding_grade[key] = "極度推薦"

    # 使用搜尋歷史資料渲染模板
    # return render_template("page2.html", data=reccomanding_grade)
    return jsonify(reccomanding_grade)


# 接平均租金
@app.route("/render_knock", methods=["POST", "GET"])
def render_knock():
    data = request.json
    address = data["address"]
    address_info = gis.buffer_analysis(address)
    address_district_info = gis.user_district(address_info)
    get_knock = gis.user_data(address_district_info)
    print(get_knock)

    for key, value in get_knock.items():
        if type(value) != int and type(value) != str:
            get_knock[key] = int(value)

    return jsonify(get_knock)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
