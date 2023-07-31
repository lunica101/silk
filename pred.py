from ultralytics import YOLO


class Yolopred:
    def __init__(self,img):
        pass
        
    def getprediction(img,InusedModel="YoloModel//silk1.pt"):
        model = YOLO(InusedModel)
        predict = model.predict(source=img , save = True , show = True)
        return predict
    
    def __del__(self):
        pass