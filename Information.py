from ultralytics import YOLO
import os
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.engine.predictor import BasePredictor

def get_yolo_result(img=None):
    if img is None:
        return {}

    model = YOLO("YoloModel//silk2_EN.pt")
    predict = model.predict(source=img, save=True, show=False, project=os.getcwd())

      #count each
    result_names = predict[0].names
    classic = predict[0].boxes.cls
    classic = [int(x.item()) for x in classic]
    count = {}
    for name in classic:
        count[result_names[name]] = count.get(result_names[name], 0) + 1
    #print(count)
    
      #accuracy
    accu = predict[0].boxes.conf
    accu = [round(float(x.item()),2) for x in accu]
    print(accu)

      #position
    #posi = predict[0].boxes.xyxy
    #posi = [t.numpy() for t in posi]
    #print(posi)

      #savedir
    leen = predict[0].save_dir
    a = os.path.basename(img)
    path_direc = os.path.join(leen,a)
    #print(path_direc)
    
    return count , accu , path_direc

get_yolo_result("C:\\Users\\USER\\Desktop\\yoloML\\ultralytics\\test\\IMG_8115.JPG") 