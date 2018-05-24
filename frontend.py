from backend import TinyYolo
import numpy as np
import tensorflow as tf
from keras.layers import Input,Conv2D,Lambda,Reshape
from keras.models import Model
from keras.utils import multi_gpu_model
from preprocessing import BatchGenerator
from keras.callbacks import EarlyStopping,ModelCheckpoint

class YOLO(object):
  def __init__(self,architecture,
               input_size,
               labels,
               max_box_per_img,
               anchors,
               gpus=1):
    self.input_size=input_size
    self.labels=list(labels)
    self.nb_class=len(self.labels)
    self.nb_box=5
    self.class_wt=np.ones(self.nb_class,dtype="float32")
    self.anchors=anchors
    self.gpus=gpus
    self.max_box_per_img=max_box_per_img

    ##Define model with cpu
    with tf.device("/cpu:0"):
      input_=Input(shape=(self.input_size,self.input_size,3))
      self.true_boxes=Input(shape=(1,1,1,self.max_box_per_img,4))

      if architecture=="Tiny Yolo":
        self.feature_extractor=TinyYolo(self.input_size)
      else:
        raise Exception("Architecture not found...")

      self.grid_h,self.grid_w=self.feature_extractor.get_output_shape()
      features=self.feature_extractor.extract(input_)

      output=Conv2D(self.nb_box*(4+1+self.nb_class),(1,1),strides=(1,1),padding="same")(features)
      output=Reshape((self.grid_h,self.grid_w,self.nb_box,4+1+self.nb_class))(output)
      output=Lambda(lambda args:args[0])([output,self.true_boxes])

      self.orgmodel=Model([input_,self.true_boxes],output)
    if gpus>1:
      self.model=multi_gpu_model(self.orgmodel,self.gpus)
    else:
      self.model=self.orgmodel

  def custom_loss(self,y_true,y_pred):
    mask_shape=tf.shape(y_true)[:4]

    cell_x=tf.to_float(
        tf.reshape(
            tf.tile(tf.range(self.grid_w),[self.grid_h]),(1,self.grid_h,self.grid_w,1,1)
            )
        )
    cell_y=tf.transpose(cell_x,(0,2,1,3,4))

    cell_grid=tf.tile(tf.concat([cell_x,cell_y],-1),[self.batch_size,1,1,self.nb_box,1])

    coord_mask=tf.zeros(mask_shape)
    conf_mask=tf.zeros(mask_shape)
    class_mask=tf.zeros(mask_shape)

    """adjust prediction"""
    pred_box_xy=tf.sigmoid(y_pred[...,:2])+cell_grid
    pred_box_wh=tf.exp(y_pred[...,2:4])*np.reshape(self.anchors,[1,1,1,self.nb_box,2])
    pred_box_conf=tf.sigmoid(y_pred[...,4])
    pred_box_class=y_pred[...,5:]

    """adjust ground truth"""
    true_box_xy=y_true[...,0:2]
    true_box_wh=y_true[...,:2:4]

    ##assign the iou area as the true confidence
    true_wh_half=true_box_wh/2.
    true_mins=true_box_xy-true_wh_half
    true_maxs=true_box_xy+true_wh_half

    pred_wh_half=pred_box_wh/2.
    pred_mins=pred_box_xy-pred_wh_half
    pred_maxs=pred_box_xy+pred_wh_half

    intersect_mins=tf.maximum(pred_mins,true_mins)
    intersect_maxs=tf.minimum(pred_maxs,true_maxs)
    intersect_wh=tf.maximum(intersect_maxs-intersect_mins,0)
    intersect_areas=intersect_wh[...,0]*intersect_wh[...,1]

    true_areas=true_box_wh[...,0]*true_box_wh[...,1]
    pred_areas=pred_box_wh[...,0]*pred_box_wh[...,1]

    union_areas=true_areas+pred_areas-intersect_areas

    iou_scores=tf.truediv(intersect_areas,union_areas)

    true_box_conf=iou_scores*y_true[...,4]

    true_box_class=tf.argmax(y_true[...,5:],-1)

    """determine the mask"""
    coord_mask=tf.expand_dims(y_true[...,4],axis=-1)*self.coord_scale

    ##assign the object and no_object penalty with ious
    true_xy=self.true_boxes[...,0:2]
    true_wh=self.true_boxes[...,2:4]

    true_wh_half=true_wh/2.
    true_mins=true_xy-true_wh_half
    true_maxs=true_xy+true_wh_half

    pred_xy=tf.expand_dims(pred_box_xy,4)
    pred_wh=tf.expand_dims(pred_box_wh,4)

    pred_wh_half=pred_wh/2.
    pred_mins=pred_xy-pred_wh_half
    pred_maxs=pred_xy+pred_wh_half

    intersect_mins=tf.maximum(pred_mins,true_mins)
    intersect_maxs=tf.minimum(pred_maxs,true_maxs)
    intersect_wh=tf.maximum(intersect_maxs-intersect_mins,0)
    intersect_areas=intersect_wh[...,0]*intersect_wh[...,1]

    pred_areas=pred_wh[...,0]*pred_wh[...,1]
    true_areas=true_wh[...,0]*true_wh[...,1]
    union_areas=pred_areas+true_areas-intersect_areas

    iou_scores=tf.truediv(intersect_areas,union_areas)
    best_iou=tf.reduce_max(iou_scores,axis=4)

    conf_mask+=tf.to_float(best_iou<0.6)*(1-y_true[...,4])*self.no_object_scale
    conf_mask+=y_true[...,4]*self.object_scale

    class_mask+=y_true[...,4]*tf.gather(self.class_wt,true_box_class)*self.class_scale

    """final loss"""
    nb_coord_box=tf.reduce_sum(tf.to_float(coord_mask>0.))
    nb_conf_box=tf.reduce_sum(tf.to_float(conf_mask>0.))
    nb_class_box=tf.reduce_sum(tf.to_float(class_mask>0.))

    loss_xy=tf.reduce_sum(coord_mask*tf.sqaure(true_box_xy-pred_box_xy))/(nb_coord_box+1e-6)/2.
    loss_wh=tf.reduce_sum(coord_mask*tf.sqaure(true_box_xy-pred_box_xy))/(nb_coord_box+1e-6)/2.
    loss_conf=tf.reduce_sum(conf_mask*tf.sqaure(true_box_conf-pred_box_conf))/(nb_conf_box+1e-6)/2.
    loss_class=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class,logits=pred_box_class)
    loss_class=tf.reduce_sum(loss_class*class_mask)/(nb_class_box+1e-6)/2.

    loss=loss_xy+loss_wh+loss_conf+loss_class
    return loss

  def train(self,train_imgs,
            valid_imgs,
            train_times,
            valid_times,
            nb_epochs,
            learning_rate,
            batch_size,
            warmup_epochs,
            object_scale,
            no_object_scale,
            coord_scale,
            class_scale,
            saved_weights_name="best_weights.h5"):
    self.batch_size=batch_size
    self.object_scale=object_scale
    self.no_object_scale=no_object_scale
    self.coord_scale=coord_scale
    self.class_scale=class_scale

    generator_config={
        "IMAGE_H":self.input_size,
        "IMAGE_W":self.input_size,
        "GRID_H":self.grid_h,
        "GRID_W":self.grid_w,
        "BOX":self.nb_box,
        "LABELS":self.labels,
        "CLASS":len(self.labels),
        "ANCHORS":self.anchors,
        "BATCH_SIZE":self.batch_size,
        "TRUE_BOX_BUFFER":self.max_box_per_img
        }

    train_generator=BatchGenerator(train_imgs,generator_config,norm=self.feature_extractor.normalize)
    
    valid_generator=BatchGenerator(valid_imgs,generator_config,norm=self.feature_extractor.normalize)
    
    self.model.compile(loss=self.custom_loss,optimizer="adam")
    
    early_stopping=EarlyStopping(monitor="val_loss",patience=5,mode="min",verbose=1)
    checkpoint=ModelCheckpoint(saved_weights_name,monitor="val_loss",verbose=1,save_best_only=True,mode="min")
    
    self.model.fit_generator(generator=train_generator,
                             steps_per_epoch=len(train_generator)*train_times,
                             epochs=nb_epochs,
                             validation_data=valid_generator,
                             validation_steps=len(valid_generator)*valid_times,
                             callbacks=[early_stopping,checkpoint])



if __name__=="__main__":
  yolo=YOLO("Tiny Yolo",416,["RBC"],10,[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828])
  print(yolo.model.summary())

