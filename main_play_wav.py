import json
from playsound import playsound
import simpleaudio as sa
import winsound

def readJson(path):
    flag_ = False
    while not flag_:
        try:
            file = open(path)
            json_data = json.load(file)
        except json.decoder.JSONDecodeError: 
            pass
        else:
            flag_ = True
    file.close()
    return json_data

def writeJson(RootDir, fileInfo):
    with open(RootDir, 'w') as write_file:
        json.dump(fileInfo, write_file, ensure_ascii=False, indent = '\t')
    write_file.close() 
    return





while True:
    resultsJsonInfo = readJson('./flag.json')
    if resultsJsonInfo['flag'] == 1:
        print('sound is played')
        filename = 'main.wav'
        winsound.PlaySound(filename, winsound.SND_FILENAME)
        
        resultsJsonInfo['flag'] = 0
        writeJson('./flag.json', resultsJsonInfo)
        print('flag is set to 0')