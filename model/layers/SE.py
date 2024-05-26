# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf


# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# https://arxiv.org/pdf/1709.01507.pdf
# --------------------------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package = 'SE')
class SqueezeAndExcitation(layers.Layer):
    

    def __init__(self, name, num_filters, expansion = 0.25):
        
        super(SqueezeAndExcitation, self).__init__()
        
        self.name_layer = name
        self.num_filters = num_filters
        self.expansion = expansion
        
        self.layers = tf.keras.Sequential([
            layers.GlobalAvgPool2D(keepdims = True, name = self.name_layer + "_se_gap_2d"),
            layers.Dense(int(num_filters * expansion), use_bias = False, activation = 'gelu', name = self.name_layer + "_se_dense_gelu"),
            layers.Dense(num_filters, use_bias = False, activation = 'sigmoid', name = self.name_layer + "_se_dense_sigmoid")
        ])


    def get_config(self):

        return {'name': self.name_layer, 'num_filters': self.num_filters, 'expansion': self.expansion}


    def call(self, inputs):
        
        x = self.layers(inputs)
        
        return x * inputs