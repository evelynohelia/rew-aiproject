
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization
from keras import backend as K
from keras import optimizers, regularizers, Model
from keras.applications import densenet

from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import os
from django.conf import settings


class ClothingPrediction:
    
    def prediction(self):
        return {"num_label_clothing": '5', 
                "label_clothing":'blusa',
                "num_tienda": '3',
                "label_tienda": 'EtaFashion',
                "porcentaje": '89%'
                }