import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from pred import Yolopred
import os

x = Yolopred.getprediction(img="C:\\Users\\USER\\Desktop\\yoloML\\ultralytics\\test\\IMG_3746.jpg",InusedModel="YoloModel//silk1.pt",path = os.getcwd())