import argparse
import os
import json
from preprocessing import parse_annotation
import numpy as np
from frontend import YOLO
from matplotlib import pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
argparser=argparse.ArgumentParser()
argparser.add_argument("-c","--conf",help="configuration file path")

def parse_config(args):
  config_path=args.conf
  with open(config_path) as config_buffer:
    config=json.loads(config_buffer.read())
  os.environ["CUDA_VISIBLE_DEVICES"]=config["env"]["gpu"]
  gpus=max(1,len(config["env"]["gpu"].split(",")))
	
  imgs,labels=parse_annotation(config["train"]["train_annot_folder"],
                               config["train"]["train_image_folder"],config["model"]["labels"])

  train_valid_split=int(0.8*len(imgs))
  np.random.shuffle(imgs)

  valid_imgs=imgs[train_valid_split:]
  train_imgs=imgs[:train_valid_split]

  overlap_labels=set(config["model"]["labels"]).intersection(set(labels.keys()))

  print("Seen labels: "+str(labels))
  print("Given labels: "+str(config["model"]["labels"]))
  print("Overelap labels: "+str(overlap_labels))
  if len(overlap_labels)<len(config["model"]["labels"]):
    print("Some labels have no image! Please check it.")
    return 
	

  #################################construct model################################
  yolo=YOLO(architecture=config["model"]["architecture"],
            input_size=config["model"]["input_size"],
            labels=config["model"]["labels"],
            max_box_per_img=config["model"]["max_box_per_image"],
            anchors=config["model"]["anchors"])

  #################################train model###################################

  [x,b],y=yolo.train(
      train_imgs,
      valid_imgs,
      config["train"]["train_times"],
      config["valid"]["valid_times"],
      config["train"]["nb_epoch"],
      config["train"]["learning_rate"],
      config["train"]["batch_size"],
      config["train"]["warmup_batches"],
      config["train"]["object_scale"],
      config["train"]["no_object_scale"],
      config["train"]["coord_scale"],
      config["train"]["class_scale"],
      saved_weights_name=config["train"]["saved_weights_name"])
  

if __name__=="__main__":
  parse_config(argparser.parse_args())
