#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 11:20:09 2018

@author: zmj
"""
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from imgaug import augmenters as iaa
from keras.utils import Sequence
from utils import BndBox,bbox_iou
import copy

def parse_annotation(anno_dir,img_dir,labels):
  all_imgs=[]
  seen_labels={}
    
  for ann in sorted(os.listdir(anno_dir)):
    img={"object":[]}
    tree=ET.parse(anno_dir+ann)
    
    for elem in tree.iter():
      if "filename" in elem.tag:
        img["filename"]=img_dir+elem.text+".jpg"
#        print(img["filename"])
      if "width" in elem.tag:
        img["width"]=int(elem.text)
#        print(img["width"])
      if "height" in elem.tag:
        img["height"]=int(elem.text)
#        print(img["height"])
      if "object" in elem.tag or "part" in elem.tag:
        obj={}
        for attr in list(elem):
          if "name" in attr.tag:
            obj["name"]=attr.text
            if obj["name"] in seen_labels:
              seen_labels[obj["name"]]+=1
            else:
              seen_labels[obj["name"]]=1
            if len(labels)>0 and obj["name"] not in labels:
              break
            else:
              img["object"]+=[obj]
          if "bndbox" in attr.tag:
            for dim in list(attr):
              if "xmin" in dim.tag:
                obj["xmin"]=int(round(float(dim.text)))
              if "ymin" in dim.tag:
                obj["ymin"]=int(round(float(dim.text)))
              if "xmax" in dim.tag:
                obj["xmax"]=int(round(float(dim.text)))
              if "ymax" in dim.tag:
                obj["ymax"]=int(round(float(dim.text)))
    if len(img["object"])>0:
      all_imgs+=[img]
  return all_imgs,seen_labels

class BatchGenerator(Sequence):
  def __init__(self,imgs,
                    config,
                    norm=None):
    self.generator=None
    self.imgs=imgs
    self.config=config
    self.norm=norm
    self.counter=0
    self.anchors=[BndBox(0,0,config["ANCHORS"][2*i],config["ANCHORS"][2*i+1]) for i in range(len(config["ANCHORS"])//2)]

    self.aug_pipe=iaa.Sequential([iaa.SomeOf((0, 5),
                      [
                          iaa.OneOf([
                              iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                              iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                              iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                          ]),
                          iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                          iaa.OneOf([
                              iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                          ]),
                          iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                          iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                          iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                      ],random_order=True)
              ],random_order=True)
    np.random.shuffle(self.imgs)

  def __len__(self):
    return int(np.ceil(len(self.imgs)/self.config["BATCH_SIZE"]))

  def __getitem__(self,idx):
    l_bound=idx*self.config["BATCH_SIZE"]
    r_bound=(idx+1)*self.config["BATCH_SIZE"]

    if r_bound>len(self.imgs):
      r_bound=len(self.imgs)
      l_bound=len(self.imgs)-self.config["BATCH_SIZE"]

    instance_count=0

    x_batch=np.zeros((r_bound-l_bound,self.config["IMAGE_H"],self.config["IMAGE_W"],3))
    b_batch=np.zeros((r_bound-l_bound,1,1,1,self.config["TRUE_BOX_BUFFER"],4))
    y_batch=np.zeros((r_bound-l_bound,self.config["GRID_H"],self.config["GRID_W"],self.config["BOX"],4+1+len(self.config["LABELS"])))

    for train_instance in self.imgs[l_bound:r_bound]:
      img,all_objs=self.aug_img(train_instance)

      true_box_index=0

      for obj in all_objs:
        center_x=0.5*(obj["xmin"]+obj["xmax"])
        center_x=center_x/(float(self.config["IMAGE_W"])/self.config["GRID_W"])
        center_y=0.5*(obj["ymin"]+obj["ymax"])
        center_y=center_y/(float(self.config["IMAGE_H"])/self.config["GRID_H"])

        grid_x=int(np.floor(center_x))
        grid_y=int(np.floor(center_y))

        obj_index=self.config["LABELS"].index(obj["name"])

        center_w=(obj["xmax"]-obj["xmin"])/(float(self.config["IMAGE_W"])/self.config["GRID_W"])
        center_h=(obj["ymax"]-obj["ymin"])/(float(self.config["IMAGE_H"])/self.config["GRID_H"])

        box=[center_x,center_y,center_w,center_h]

        best_anchor=-1
        max_iou=-1
        shifted_box=BndBox(0,0,center_w,center_h)
        for i in range(len(self.anchors)):
          anchor=self.anchors[i]
          iou=bbox_iou(shifted_box,anchor)

          if max_iou<iou:
            best_anchor=i
            max_iou=iou

        y_batch[instance_count,grid_y,grid_x,best_anchor,0:4]=box
        y_batch[instance_count,grid_y,grid_x,best_anchor,4]=1
        y_batch[instance_count,grid_y,grid_x,best_anchor,5+obj_index]=1

        b_batch[instance_count,0,0,0,true_box_index]=box

        true_box_index+=1
        true_box_index=true_box_index%self.config["TRUE_BOX_BUFFER"]
      if self.norm!=None:
        x_batch[instance_count]=self.norm(img)
      else:
        for obj in all_objs:
          cv2.rectangle(img[:,:,::-1],(obj["xmin"],obj["ymin"]),(obj["xmax"],obj["ymax"]),(255,0,0),3)
          cv2.putText(img[:,:,::-1],obj["name"],(obj["xmin"]+2,obj["ymin"]+12),0,1.2e-3*img.shape[0],(0,255,0),2)
          x_batch[instance_count]=img
      instance_count+=1
    return [x_batch,b_batch],y_batch


  def aug_img(self,train_instance):
    #print(train_instance)
    img_name=train_instance["filename"]
    img=cv2.imread(img_name)

    h,w,c=img.shape
    all_objs=copy.deepcopy([train_instance["object"]])[0]

    img=self.aug_pipe.augment_image(img)

    img=cv2.resize(img,(self.config["IMAGE_H"],self.config["IMAGE_W"]))
    img=img[:,:,::-1]##BGR-->RGB

    for obj in all_objs:
      for attr in ["xmin","xmax"]:
        obj[attr]=int(obj[attr]*float(self.config["IMAGE_W"])/w)
      for attr in ["ymin","ymax"]:
        obj[attr]=int(obj[attr]*float(self.config["IMAGE_H"])/h)
    return img,all_objs


if __name__=="__main__":
  anno_dir="../dataset-master/Annotations/"
  img_dir="../dataset-master/JPEGImages/"
  labels=["RBC"]
  imgs,labels=parse_annotation(anno_dir,img_dir,labels)
