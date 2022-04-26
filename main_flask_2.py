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

def readJson(path):
    file = open(path)
    json_data = json.load(file)
    file.close()
    return json_data

def writeJson(RootDir, fileInfo):
    with open(RootDir, 'w') as write_file:
        json.dump(fileInfo, write_file, ensure_ascii=False, indent = '\t')
    write_file.close() 
    return

save_dir = './'

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
    # Emotion = model_sanghoon(Text)
    Emotion = 'Angry'
    print('TER is done')
    ################################## 
    
    # ESS
    ##################################
    print('Text',Text)
    wav = main_ess.run(Text, Emotion)
    print('ESS is done')
    ################################## 
    
    # wav 저장 및 GUI 에서 스피커 출력
    ################################## 
    # wav 저장
    Fs = 22050

    # try:
    #     os.remove("main.wav")
    # except FileNotFoundError:
    #     pass
            
    scipy.io.wavfile.write("main.wav", Fs, wav)
    print('wav is saved')

    # 스피커 출력을 위한 flag 설정
    resultsJsonInfo = readJson('./flag.json')
    if resultsJsonInfo['flag'] == 0:
        resultsJsonInfo['flag'] = 1
        writeJson('./flag.json', resultsJsonInfo)
        print('flag is set to 1')
    # if os.path.isfile('./flag.json'):
    #     resultsJsonInfo = readJson('./flag.json')
    #     resultsJsonInfo['flag'] = 1
    #     print('1')
    # else:
    #     resultsJsonInfo = {}
    #     resultsJsonInfo['flag'] = 1
    #     print('2')
        
    # writeJson('./flag.json', resultsJsonInfo)
    # print('flag is set to 1')
    ################################## 

    return Emotion

if __name__ == '__main__':

    app.run(host=IP, port=PORT, debug=True)