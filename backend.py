from keras.layers import Input,BatchNormalization,LeakyReLU,Conv2D,MaxPooling2D
from keras.models import Model


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
		x=Conv2D(16,(3,3),padding="same",use_bias=False)(input_)
		x=BatchNormalization()(x)
		x=LeakyReLU(alpha=0.1)(x)
		x=MaxPooling2D()(x)

		#Layer 2-5
		for i in range(4):
			x=Conv2D(32*(2**i),(3,3),padding="same",use_bias=False)(x)
			x=BatchNormalization()(x)
			x=LeakyReLU(alpha=0.1)(x)
			x=MaxPooling2D()(x)

		#Layer 6
		x=Conv2D(512,(3,3),padding="same",use_bias=False)(x)
		x=BatchNormalization()(x)
		x=LeakyReLU(alpha=0.1)(x)
		x=MaxPooling2D(strides=(1,1),padding="same")(x)

		#Layer 7-8
		for i in range(2):
			x=Conv2D(1024,(3,3),padding="same",use_bias=False)(x)
			x=BatchNormalization()(x)
			x=LeakyReLU(alpha=0.1)(x)

		self.feature_extractor=Model(input_,x)
	def normalize(self,img):
		return img/255.

if __name__=="__main__":
	model=TinyYolo(416)
	print(model.feature_extractor.summary())