# -*- coding: utf-8 -*-
"""
Created on Sat May 26 18:18:47 2018

@author: ZMJ
"""
from frontend import YOLO
import cv2
import argparse
import os
import json
from utils import draw_boxes
from matplotlib import pyplot as plt
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
argparser=argparse.ArgumentParser()
argparser.add_argument("-c","--conf",help="configuration file path")
argparser.add_argument("-w","--weights",help="weights file path")

def test(argparser,test_path):
  args=argparser.parse_args()
  config_path=args.conf
  weights_path=args.weights
  with open(config_path) as config_buffer:
    config=json.loads(config_buffer.read())
  os.environ["CUDA_VISIBLE_DEVICES"]=config["env"]["gpu"]
  
  yolo=YOLO(architecture=config["model"]["architecture"],
            input_size=config["model"]["input_size"],
            labels=config["model"]["labels"],
            max_box_per_img=config["model"]["max_box_per_image"],
            anchors=config["model"]["anchors"])
  yolo.load_weights(weights_path)
  if test_path[-3:]=="mp4":
    pass
  else:
    for f in os.listdir(test_path):
      print(f)
      f_path=os.path.join(test_path,f)
#      print(f_path)
      img=cv2.imread(f_path)
      boxes=yolo.predict(img)
      img=draw_boxes(boxes,img,config["model"]["labels"])
      cv2.imwrite(f_path[-9:],img)
      break
#      plt.imshow(img)
#      plt.show()

test(argparser,test_path="RBC_datasets/JPEGImages/")
