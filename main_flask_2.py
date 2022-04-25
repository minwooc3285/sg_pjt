# -*- coding: utf-8 -*-
# Final Debugged by ysHeo at 220209_2126

import os
from pprint import pprint
import random  
import pandas as pd
import numpy as np
#from mf import MatrixFactorization
from flask import Flask

import scipy.io.wavfile
import json
import requests
import main_ess

save_dir = './'

TOPIC_NAME = "movielog24"
KAFKA_SERVER = "localhost:9092"
IP = "0.0.0.0"
PORT = 8082

app = Flask(__name__)
@app.route('/')
@app.route('/home')
def home():
    return 'Welcome'

@app.route('/message/<Text>') 
def message(Text):

    # txt 를 TER
    ##################################
    Emotion = model_sanghoon(Text)
    ################################## 
    
    # ESS
    ##################################
    wav = main_ess.run(Text, Emotion)
    ################################## 
    
    # wav 저장 및 GUI 에서 스피커 출력
    ################################## 
    # wav 저장
    Fs = 22050
    scipy.io.wavfile.write("respond.wav", Fs, wav)
    
    # GUI 에서 스피커 출력
    
    ################################## 

    return wav

if __name__ == '__main__':

    app.run(host=IP, port=PORT, debug=True)