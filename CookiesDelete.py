import time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import os

local_path = os.getcwd()
EXTENSION_FILE = local_path + r"\istilldontcareaboutcookies-1.1.8.xpi" #Filen som indeholder extensionen som blokerer cookies
options = Options()
driver = webdriver.Firefox(options=options)
driver.install_addon(EXTENSION_FILE, temporary=True)
time.sleep(3)
driver.get("https://www.bbc.com/news/world/europe")
time.sleep(5)

