from keras.layers import Input,BatchNormalization,Conv2D,MaxPooling2D,Lambda,concatenate,LeakyReLU
from keras.models import Model
import tensorflow as tf

ALPHA=1.0
TINY_YOLO_WEIGHTS="models/tiny_yolo_backend.h5"
FULL_YOLO_BACKEND_PATH="models/full_yolo_backend.h5"


class BaseFeatureExtractor(object):
  def __init__(self,input_size):
    raise NotImplementedError("intialized failed...")

  def normalize(self,img):
    raise NotImplementedError("normalized failed...")

  def get_output_shape(self):
    return self.feature_extractor.get_output_shape_at(-1)[1:3]

  def extract(self,input_img):
    return self.feature_extractor(input_img)

class TinyYolo(BaseFeatureExtractor):
  def __init__(self,input_size):
    input_=Input(shape=(input_size,input_size,3))
    
    #Layer 1
    x=Conv2D(int(ALPHA*16),(3,3),padding="same",use_bias=False)(input_)
    x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=MaxPooling2D()(x)

    #Layer 2-5
    for i in range(4):
      x=Conv2D(int(ALPHA*32*(2**i)),(3,3),padding="same",use_bias=False)(x)
      x=BatchNormalization()(x)
      x=LeakyReLU(alpha=0.1)(x)
      x=MaxPooling2D()(x)

    #Layer 6
    x=Conv2D(int(ALPHA*512),(3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=MaxPooling2D(strides=(1,1),padding="same")(x)

    #Layer 7-8
    for i in range(2):
      x=Conv2D(int(ALPHA*1024),(3,3),padding="same",use_bias=False)(x)
      x=BatchNormalization()(x)
      x=LeakyReLU(alpha=0.1)(x)
    
    self.feature_extractor=Model(input_,x)
    self.feature_extractor.load_weights(TINY_YOLO_WEIGHTS)
    print("load weights from "+TINY_YOLO_WEIGHTS)
  def normalize(self,img):
    return img/255.
class FullYolo(BaseFeatureExtractor):
  def __init__(self, input_size):
    input_image = Input(shape=(input_size, input_size, 3))
    
    my_lrelu_layer=Lambda(lrelu,output_shape=lrelu_outshape)
    def space_to_depth_x2(x):
      return tf.space_to_depth(x, block_size=2)

    x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
    x = BatchNormalization(name='norm_1')(x)
    x = my_lrelu_layer(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = my_lrelu_layer(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = my_lrelu_layer(x)

    x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = my_lrelu_layer(x)

    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = my_lrelu_layer(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = my_lrelu_layer(x)

    x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = my_lrelu_layer(x)

    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
    x = BatchNormalization(name='norm_8')(x)
    x = my_lrelu_layer(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
    x = BatchNormalization(name='norm_9')(x)
    x = my_lrelu_layer(x)

    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
    x = BatchNormalization(name='norm_10')(x)
    x = my_lrelu_layer(x)

    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
    x = BatchNormalization(name='norm_11')(x)
    x = my_lrelu_layer(x)

    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
    x = BatchNormalization(name='norm_12')(x)
    x = my_lrelu_layer(x)

    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = my_lrelu_layer(x)

    skip_connection = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = my_lrelu_layer(x)

    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = my_lrelu_layer(x)

    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = my_lrelu_layer(x)
    
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = my_lrelu_layer(x)

    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
    x = BatchNormalization(name='norm_18')(x)
    x = my_lrelu_layer(x)

    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
    x = BatchNormalization(name='norm_19')(x)
    x = my_lrelu_layer(x)

    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
    x = BatchNormalization(name='norm_20')(x)
    x = my_lrelu_layer(x)

    skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
    skip_connection = BatchNormalization(name='norm_21')(skip_connection)
    skip_connection = my_lrelu_layer(skip_connection)
    skip_connection = Lambda(space_to_depth_x2)(skip_connection)

    x = concatenate([skip_connection, x])

    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
    x = BatchNormalization(name='norm_22')(x)
    x = my_lrelu_layer(x)

    self.feature_extractor = Model(input_image, x)  
#    self.feature_extractor.load_weights(FULL_YOLO_BACKEND_PATH)

  def normalize(self, image):
    return image / 255.
if __name__=="__main__":
  model=TinyYolo(416)
  print(model.feature_extractor.summary())
