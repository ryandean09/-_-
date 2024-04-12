# 操作 browser 的 API
from selenium.webdriver.chrome.service import Service
from selenium import webdriver

# 處理逾時例外的工具
from selenium.common.exceptions import TimeoutException

# 面對動態網頁，等待某個元素出現的工具，通常與 exptected_conditions 搭配
from selenium.webdriver.support.ui import WebDriverWait

# 搭配 WebDriverWait 使用，對元素狀態的一種期待條件，若條件發生，則等待結束，往下一行執行
from selenium.webdriver.support import expected_conditions as EC

# 期待元素出現要透過什麼方式指定，通常與 EC、WebDriverWait 一起使用
from selenium.webdriver.common.by import By

# 強制等待 (執行期間休息一下)
from time import sleep

import csv

# send_keys模擬鍵盤輸入
from selenium.webdriver.common.keys import Keys

import random

import re

# listening
import pyautogui
from time import sleep, time
import keyboard
from IPython.display import clear_output

# 多執行緒
import threading



def initialize_browser():
    # 啟動瀏覽器工具的選項
    my_options = webdriver.ChromeOptions()
    # my_options.add_argument("--headless")                #不開啟實體瀏覽器背景執行
    my_options.add_argument("--start-maximized")         #最大化視窗
    my_options.add_argument("--incognito")               #開啟無痕模式
    my_options.add_argument("--disable-popup-blocking") #禁用彈出攔截
    my_options.add_argument("--disable-notifications")  #取消 chrome 推播通知
    my_options.add_argument("--lang=zh-TW")  #設定為正體中文

    # 使用 Chrome 的 WebDriver
    driver = webdriver.Chrome(
        options = my_options
    )

    return driver


def scraping(driver):
    with open(r'./name.txt', 'r', encoding='utf-8') as file:
        dist_list = file.readlines()
        dataset = [['name', 'class', 'address', 'latitude', 'longitude']]
        
        for search_name in dist_list:
            search_name = search_name.replace('\n','')
            
            # 前往google_map網站
            driver.get('https://www.google.com/maps?authuser=0')
        
            # 尋找網頁中的搜尋框
            input_block = driver.find_element(
                By.CSS_SELECTOR,
                'input.searchboxinput.xiQnY'
            )
            print(f'{search_name}-starting...')
            # 在搜尋框中輸入文字
            input_block.send_keys(f'{search_name}')
            sleep(0.1)
            input_block.send_keys(Keys.ENTER)
            sleep(0.5)
        
            try:
                # 取得google_map基本內容
                element = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "div.lMbq3e"))
                            )
                
                sleep(2)
            
                # 車站名稱
                name = element.find_element(
                    By.CSS_SELECTOR,
                    'h1.DUwDvf.lfPIob'
                )
            
            
                # 設施種類
                class_ = element.find_element(
                    By.CSS_SELECTOR,
                    'button.DkEaL '
                )
            
            
                # 車站地址
                address = driver.find_element(
                    By.CSS_SELECTOR,
                    'div.Io6YTe.fontBodyMedium.kR99db'
                )
            
                regex_address = r'[^0-9].*'  # 106台北市大安區復興南路二段206號 => 台北市大安區復興南路二段206號
                search_address = re.search(regex_address, address.text)
            
            
                # coordinate(經緯度)
                link = driver.current_url # 當前網頁網址
                regex_coordinate = r".\w.\w+!4d.\w+.\w+"  # link => 25.0266645!4d121.5431999
                search_coordinate = re.search(regex_coordinate, link)
                # 切分成緯度、經度
                coordinate = search_coordinate[0].split('!4d')  # coordinate:座標(經緯度) 
            
                dataset.append([name.text, class_.text, search_address[0], coordinate[0], coordinate[1]])
                print(123)

                    
            except:  
                try:
                    search_block = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "a.hfpxzc"))
                            )
                                
                    search_block.click()

                    element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "div.lMbq3e"))
                        )
                        

                    sleep(2)
    
                    # 車站名稱
                    name = element.find_element(
                        By.CSS_SELECTOR,
                        'h1.DUwDvf.lfPIob'
                    )
                
                
                    # 設施種類
                    class_ = element.find_element(
                        By.CSS_SELECTOR,
                        'button.DkEaL '
                    )
                
        
                    # 車站地址
                    address = driver.find_element(
                        By.CSS_SELECTOR,
                        'div.Io6YTe.fontBodyMedium.kR99db'
                    )
                
                    regex_address = r'[^0-9].*'  # 106台北市大安區復興南路二段206號 => 台北市大安區復興南路二段206號
                    search_address = re.search(regex_address, address.text)
                
                
                    # coordinate(經緯度)
                    link = driver.current_url # 當前網頁網址
                    regex_coordinate = r".\w.\w+!4d.\w+.\w+"  # link => 25.0266645!4d121.5431999
                    search_coordinate = re.search(regex_coordinate, link)
                    # 切分成緯度、經度
                    coordinate = search_coordinate[0].split('!4d')  # coordinate:座標(經緯度) 
                
                    dataset.append([name.text, class_.text, search_address[0], coordinate[0], coordinate[1]])
                    print(1234)
                except:
                    print(search_name + 'null')
                    continue
        return dataset 
                


    
    

def to_csv(dataset):
    # 將 list 存成 csv
    with open(f"./dataset.csv", "w", newline='', encoding="utf-8-sig") as file:
        w = csv.writer(file)
        # 寫入二維表格
        w.writerows(dataset)
    #    print(f'{search_name}-ending...')

def listening():
    global switch

    while switch:
        # 睡一下
        sleep(0.01)

        #自動清除此格的文字輸出
        clear_output(wait=True)

        try:
            # 取得 Box 物件
            btn_reload = pyautogui.locateOnScreen(
                './reload.png',
                confidence=0.9 # opencv信心指數，用 1 的話圖片一點灰塵污漬都不能有 
            )

            btn_reload_point = pyautogui.center(btn_reload)

            # 按下 重新載入
            pyautogui.click(btn_reload_point.x, btn_reload_point.y)
        except:
            pass

if __name__ == "__main__":
    driver = initialize_browser()
    switch = True
    listening_thread = threading.Thread(target=listening) # 建立listening執行緒
    listening_thread.start() # 啟動執行緒


    dataset = scraping(driver)
    to_csv(dataset)
    
    switch = False # 爬蟲結束，結束迴圈
    listening_thread.join() # 结束執行緒
    
    driver.quit()