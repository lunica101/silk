from ultralytics import YOLO
import os

class Yolopred:        
    def getprediction(img,InusedModel="YoloModel//silk1.pt",path = os.getcwd()):
        model = YOLO(InusedModel)
        predict = model.predict(source=img , save = True , show = True , project = path )
        return predict