import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from pred import Yolopred
import os

Yolopred.getprediction(img="C:\\Users\\USER\\Desktop\\Silky\\test\\IMG_3682.jpg",InusedModel="YoloModel//silk1.pt")