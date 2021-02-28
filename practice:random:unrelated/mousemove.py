#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:26:26 2020

@author: owlthekasra
"""



from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import time

def find_element_by_id(ID):
    elem = driver.find_element_by_xpath('//*[@id={}]'.format(ID))
    return elem

def return_id_xpath(ID):
    return '//*[@id={}]'.format(ID)

url = 'https://eegedu.com/'
chromedriver = '/usr/local/bin/chromedriver'


driver = webdriver.Chrome(chromedriver)
driver.get(url)
CHOOSE_MODULE_ID = "PolarisSelect1"
RAW_AND_FILTERED_XPATH = "//*[@id='PolarisSelect1']/option[@value='4. Raw and Filtered Data']"
elem = driver.find_element_by_id(CHOOSE_MODULE_ID)
elem2 =driver.find_element_by_xpath(RAW_AND_FILTERED_XPATH)
actions = ActionChains(driver)
actions.click(elem).perform()
time.sleep(.5)
actions.click(elem2).perform()





from pynput.mouse import Button, Controller
mouse = Controller()
mouse.position

mouse.position = (788.84375, 969.24609375)

mouse.click(Button.left, 1)

mouse.move(20, -13)
mouse.click(Button, 2)
mouse.move(-150, 140)
mouse.click(Button.right, 1)
mouse.click(Button.left, 1)
mouse.press(Button.right)
mouse.release(Button.right)
mouse.scroll(0,2)

