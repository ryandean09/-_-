# 操作 browser 的 API
import csv
import random
import re
# 多執行緒
import threading
# 強制等待 (執行期間休息一下)
from time import sleep, time

# listening
import pyautogui
from IPython.display import clear_output
from selenium import webdriver
# 處理逾時例外的工具
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
# 期待元素出現要透過什麼方式指定，通常與 EC、WebDriverWait 一起使用
from selenium.webdriver.common.by import By
# send_keys模擬鍵盤輸入
from selenium.webdriver.common.keys import Keys
# 搭配 WebDriverWait 使用，對元素狀態的一種期待條件，若條件發生，則等待結束，往下一行執行
from selenium.webdriver.support import expected_conditions as EC
# 面對動態網頁，等待某個元素出現的工具，通常與 exptected_conditions 搭配
from selenium.webdriver.support.ui import WebDriverWait


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


def input_name(driver, search_name):
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


def scroll(driver):
    try:
        offset = 0
        count = 0
        start = True

        # google_map，JS動態變化前標籤
        inner_frame = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd > div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd"))
        )

        offset = driver.execute_script(
            '''return document.querySelector('div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd > div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd').scrollHeight;'''
        )

        # 使用 JavaScript 來捲動內部視窗的滾動條
        driver.execute_script(f'''
                document.querySelector("div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd > div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd").scrollTo({{
                    top: {offset},
                    behavior: 'smooth'
                }});
            ''')

        sleep(1)

        while start:

            offset = driver.execute_script(
                "return document.querySelector('div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd.QjC7t').scrollHeight;"
            )

            # google_map，JS動態變化後標籤
            inner_frame = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd.QjC7t"))
            )

            # 使用 JavaScript 來捲動內部視窗的滾動條
            driver.execute_script(f'''
                    document.querySelector("div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd.QjC7t").scrollTo({{
                        top: {offset},
                        behavior: 'smooth'
                    }});
                ''')

            # 強制休息
            sleep(random.randrange(4))

            inner_height = driver.execute_script(
                '''return document.querySelector('div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd > div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd').scrollHeight;'''
            )


            # 網頁滾到最底，結束程式
            if inner_height == offset:
                if count in [4, 6]:
                                    
                    # 往回滾動，重新loading...
                    offset = -20
                    driver.execute_script(f'''
                        document.querySelector("div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd.QjC7t").scrollBy({{
                            top: {offset},
                            behavior: 'smooth'
                            }});
                        ''')
                    
                    sleep(3)
                count += 1
            else:
                count = 0

            if count == 10:
                start = False
    except Exception as e:
        print(e)


def scraping(driver):
    # 放置爬取的資料
    dataset = [['id', 'name', 'star', 'comment', 'class', 'coordinate', 'address']]
    
    try:
        # 取得google_map基本內容
        elements = driver.find_elements(
            By.CSS_SELECTOR,
            'div.UaQhfb.fontBodyMedium'
        )

        id = 0  # 店家id

        for elm in elements:
            # 店家id
            id += 1

            # 店家名稱
            name = elm.find_element(
                By.CSS_SELECTOR,
                'div.qBF1Pd.fontHeadlineSmall'
            )

            # 店家種類
            shop_class = elm.find_elements(
                By.CSS_SELECTOR,
                'div.W4Efsd > div.W4Efsd > span'
            )

            # 有些店家沒有評論數、星數，直接給數值為0
            no_commend = elm.find_element(
                By.CSS_SELECTOR,
                'span.e4rVHe.fontBodyMedium'
            )

            if no_commend.text == '沒有評論':
                dataset.append([id, name.text, 0, 0, shop_class[0].text])
                continue

            # 星數
            star = elm.find_element(
                By.CSS_SELECTOR,
                'span.MW4etd'
            )

            # 評論數
            comment_num = elm.find_element(
                By.CSS_SELECTOR,
                'span.UY7F9'
            )

            regex_comment = r"\w+"  # (60) => 60
            comment_num_re = re.search(regex_comment, comment_num.text)

            # 資料匯入list
            dataset.append([id, name.text, eval(star.text), eval(comment_num_re[0]), shop_class[0].text])
    except Exception as e:
        print('google_map_error')


    # 取得各店家連結、經緯度    
    list_links = [] # 存放店家連結
    
    try:
        links = driver.find_elements(
            By.CSS_SELECTOR,
            'a.hfpxzc'
        )

        id = 0

        for lelm in links:
            # 店家連結
            id += 1
            link = lelm.get_attribute('href')
            list_links.append(link)

            # coordinate(經緯度)
            regex_coordinate = r"(?<=!3d).*(?=!16s)"  # link => 25.0266645!4d121.5431999
            search_coordinate = re.search(regex_coordinate, link)
            
            if search_coordinate:
                # 經緯度加入list
                coordinate = search_coordinate[0].replace('!4d', ',')  # coordinate:座標(經緯度)     
                dataset[id].append(coordinate)
            else:
                dataset[id].append('沒有座標')
        
    except Exception as e:
        print('No_links_and_coordinate')


    # 進入店家連結取得詳細資料
    id = 0
    for url in list_links:
        try:
            driver.get(url)

            # 強制休息
            sleep(random.randrange(4))
            
            id += 1

            # 詳細地址
            # address = driver.find_element(
            #     By.CSS_SELECTOR,
            #     'div.Io6YTe.fontBodyMedium.kR99db '
            # )
            address = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.Io6YTe.fontBodyMedium.kR99db"))
            )

            regex_address = r'[^0-9].*'  # 106台北市大安區復興南路二段206號 => 台北市大安區復興南路二段206號
            search_address = re.search(regex_address, address.text)
            
            if search_address:
                # 加入list
                dataset[id].append(search_address[0])
            else:
                dataset[id].append('沒有地址')
        
        except Exception as e:
            dataset[id].append('沒有地址')
            print(f'{id},No_address')
            continue
    
    return dataset

    
    

def to_csv(search_name, dataset):
    # 將 list 存成 csv
    with open(f"./{search_name}.csv", "w", newline='', encoding="utf-8-sig") as file:
        w = csv.writer(file)
        # 寫入二維表格
        w.writerows(dataset)
        print(f'{search_name}-ending...')

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

    with open(r'./dist.txt', 'r', encoding='utf-8') as file:
        dist_list = file.readlines()  # 行政區
        dist = dist_list[0].replace('\n','') # 台北市大安區
        
        for _ in range(1,len(dist_list)):
            search_name = dist_list[_].replace('\n','') # 台北市大安區莊敬里
            input_name(driver, search_name)
            scroll(driver)
            dataset = scraping(driver)
            to_csv(search_name, dataset)
            
    switch = False # 爬蟲結束，結束迴圈
    listening_thread.join() # 结束執行緒
    
    driver.quit()