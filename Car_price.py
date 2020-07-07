try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
import requests  
from bs4 import BeautifulSoup, SoupStrainer
import socks
import socket
from fake_useragent import UserAgent
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import time
from urllib.request import *
import lxml.html as html
from lxml import etree
import re

def find_urls(url, session = requests.Session()):    
    r = session.get(url, headers={'User-Agent': UserAgent().chrome})
    strainer = SoupStrainer(['a'],{'class':['Link ListingItemTitle-module__link']})
    soup = BeautifulSoup(r.text, 'lxml', parse_only=strainer)
    links = soup.find_all('a', {'class': 'Link ListingItemTitle-module__link'})
    list_link = []
    for link in links:
        link = link.get('href')
        list_link.append(link)
    return list_link
def parsing(url, session):
    m = [np.nan for k in range(24)]
    r = session.get(url, headers={'User-Agent': UserAgent().chrome})
    r.encoding = 'utf8'
    strainer = SoupStrainer(['span','div','class','a', 'ul'],
                            {'class':['Link Link_color_gray CardBreadcrumbs__itemText',
                                      'OfferPriceCaption__price',
                                      'CardInfo',
                                      'CardDescription CardOfferBody__contentIsland',
                                      'PriceNewOffer__originalPrice',
                                      'CardInfoGrouped__list']})
    soup = BeautifulSoup(r.text, 'lxml', parse_only=strainer)
    
    try:
        model = soup.find_all(class_= 'Link Link_color_gray CardBreadcrumbs__itemText')
        m[5] = model[-1].text.replace('\xa0', '')
#        print(model[-1].text)
#        print(re.search(r'\/\"\>(.*?)\<\!', str(model[-1])).group(1))
#        print(re.search(r'\/\"\>(.*?)\<\!', str(model[2])).group(1))
    except:
        pass
    
    try:
        information2 = soup.find_all(class_= 'CardInfo__cell')
        m[0] = information2[5].text
        m[1] = 'BMW'
        m[2] = information2[7].text
        m[3] = information2[9].text.replace('\xa0', '')
        m[7] = information2[1].text
        m[9] = information2[13].text        
        m[13] = information2[3].text[:-3].replace('\xa0', '')
        m[15] = information2[15].text
        m[17] = information2[19].text
        m[18] = information2[21].text.replace('\xa0', ' ')
        m[19] = information2[23].text
        m[20] = information2[25].text
#        i2 = 0
#        for info in information2:
#            print(i2, info.text)        
#            #print(i2, re.search(r'\>(.*?)\<', str(info)).group(1))
#            i2 += 1
        price = soup.find(class_= 'OfferPriceCaption__price').text
        m[23] = price[:-2].replace('\xa0', '')
#        print(price[:-2].replace('\xa0', ''))
    except:
        pass
    try:
        describe = soup.find_all(class_= 'CardDescription__textInner')
        description = ''
        for descr in describe:
            description = description + descr.text + ' '
        m[12] = description
    except:
        pass
    return m


from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool

pool = ThreadPool(10)



s = requests.Session()
for i in range(1, 100): 
    url = f'https://auto.ru/moskva/cars/bmw/used/?output_type=table&sort=year-asc&page={i}'
    list_urls = find_urls(url, s)
    results = list(pool.map(lambda x: parsing(x, s), list_urls))
    df_my = pd.DataFrame(results, columns = ['bodyType', 'brand', 'color', 'fuelType', 'modelDate',
                                             'name', 'numberOfDoors', 'productionDate', 'vehicleConfiguration',
                                             'vehicleTransmission', 'engineDisplacement', 'enginePower', 'description',
                                             'mileage', 'Комплектация', 'Привод', 'Руль', 'Состояние', 'Владельцы', 
                                             'ПТС', 'Таможня', 'Владение', 'id', 'Price'])
    df_my.to_csv('Parcing/my_parce_'+str(i)+'.csv', index = False)

for i in range(1, 100):        
    url = f'https://auto.ru/moskva/cars/bmw/used/?output_type=table&sort=year-desc&page={i}'
    list_urls = find_urls(url, s)
    results = list(pool.map(lambda x: parsing(x, s), list_urls))
    df_my = pd.DataFrame(results, columns = ['bodyType', 'brand', 'color', 'fuelType', 'modelDate',
                                             'name', 'numberOfDoors', 'productionDate', 'vehicleConfiguration',
                                             'vehicleTransmission', 'engineDisplacement', 'enginePower', 'description',
                                             'mileage', 'Комплектация', 'Привод', 'Руль', 'Состояние', 'Владельцы', 
                                             'ПТС', 'Таможня', 'Владение', 'id', 'Price'])
    df_my.to_csv('Parcing/my_parce_'+str(100 + i)+'.csv', index = False)
pool.close()
pool.join()    

