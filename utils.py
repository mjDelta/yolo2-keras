import numpy as np
import cv2

class BndBox:	
  def __init__(self,x,y,w,h,c=None,classes=None):
    self.x=x
    self.y=y
    self.w=w
    self.h=h

    self.c=c
    self.classes=classes

    self.label=-1
    self.score=-1

  def get_label(self):
    if self.label==-1:
      self.label=np.argmax(self.classes)
    return self.label

  def get_score(self):
    if self.score==-1:
      self.score=self.classes[self.get_label()]
    return self.score

def bbox_iou(box1,box2):
  xmin1=box1.x-box1.w/2
  xmax1=box1.x+box1.w/2
  ymin1=box1.y-box1.h/2
  ymax1=box1.y+box1.h/2

  xmin2=box2.x-box2.w/2
  xmax2=box2.x+box2.w/2
  ymin2=box2.y-box2.h/2
  ymax2=box2.y+box2.h/2

  intersect_w=interval_overlap([xmin1,xmax1],[xmin2,xmax2])
  intersect_h=interval_overlap([ymin1,ymax1],[ymin2,ymax2])

  intersect=intersect_w*intersect_h

  union=box1.w*box1.h+box2.w*box2.h-intersect

  return float(intersect)/union

def interval_overlap(args1,args2):
  min1,max1=args1
  min2,max2=args2
  if max1<min2 or max2<min1:
    return 0
  else:
    return min(max1,max2)-max(min1,min2)
  
def sigmoid(x):
  return 1./(1+np.exp(-x))

def softmax(x,axis=-1,t=-100):
  x=x-np.max(x)
  if np.min(x)<t:
    x=x/np.min(x)*t
  e_x=np.exp(x)

  return e_x/e_x.sum(axis,keepdims=True)

def decode_netout(netout,obj_th,nms_th,anchors,nb_class):
  grid_h,grid_w,nb_box=netout.shape[:3]
  boxes=[]
  ##decode the output
  netout[...,4]=sigmoid(netout[...,4])
  netout[...,5:]=netout[...,4][...,np.newaxis]*softmax(netout[...,5:])
#  netout[...,5:]=softmax(netout[...,5:]*netout[...,4][...,np.newaxis])
  netout[...,5:]*=netout[...,5:]>obj_th
  
  for row in range(grid_h):
    for col in range(grid_w):
      for b in range(nb_box):
        classes=netout[row,col,b,5:]
        print(np.sum(classes))
        if np.sum(classes)>0:
          x,y,w,h=netout[row,col,b,:4]
          
          x=(col+sigmoid(x))/grid_w
          y=(row+sigmoid(y))/grid_h
          w=anchors[2*b]*np.exp(w)/grid_w
          h=anchors[2*b+1]*np.exp(h)/grid_h
          confidence=netout[row,col,b,4]
          
          box=BndBox(x,y,w,h,confidence,classes)
          boxes.append(box)
#  print(len(boxes))
  for c in range(nb_class):
    sorted_indices=list(reversed(np.argsort([box.classes[c] for box in boxes])))
    
    for i in range(len(sorted_indices)):
      index_i=sorted_indices[i]
      if boxes[index_i].classes[c]==0:
        continue
      else:
        for j in range(i+1,len(sorted_indices)):
          index_j=sorted_indices[j]
          
          if bbox_iou(boxes[index_i],boxes[index_j])>=nms_th:
            boxes[index_j].classes[c]=0
  boxes=[box for box in boxes if box.get_score()>obj_th]
  return boxes

def draw_boxes(boxes,img,labels):
  h,w,c=img.shape
#  print(img.shape)
  for box in boxes:
    xmin=box.x-box.w/2;xmin=int(xmin*w)
    xmax=box.x+box.w/2;xmax=int(xmax*w)
    ymin=box.y-box.h/2;ymin=int(ymin*h)
    ymax=box.y+box.h/2;ymax=int(ymax*h)
#    print(box.x,box.y,box.w,box.h)
#    print(xmin,xmax,ymin,ymax)
    cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),2)
    cv2.putText(img,
                labels[box.get_label()]+" "+str(box.get_score()),
                (xmin,ymin-13),
                cv2.FONT_HERSHEY_SIMPLEX,
                1e-3*h,
                (255,0,0),
                2)
  return img

if __name__=="__main__":
  a=BndBox(30,30,40,40)
  b=BndBox(45,45,50,50)
  print(bbox_iou(a,b))
