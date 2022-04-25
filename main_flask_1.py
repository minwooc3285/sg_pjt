# -*- coding: utf-8 -*-
# Final Debugged by ysHeo at 220209_2126

import os
from pprint import pprint
import random  
import pandas as pd
import numpy as np
#from mf import MatrixFactorization
from flask import Flask

import json
import requests

IP = "0.0.0.0"
PORT = 8082

app = Flask(__name__)
@app.route('/')
@app.route('/home')
def home():
    return 'Welcome'

@app.route('/message/<txt>') 
def message(txt):

    # txt 를 GUI 에 출력하기 위한 구문
    ##################################
    
    ################################## 

    return txt

if __name__ == '__main__':

    app.run(host=IP, port=PORT, debug=True)