# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


from tensorflow.keras import layers
from tensorflow import keras
from .CBAM import CBAM
from .APS import CircularPad
import tensorflow as tf
import tensorflow_probability as tfp

# --------------------------------------------------------------------------------------------
# SETS THE SEED OF THE RANDOM NUMBER GENERATOR
# --------------------------------------------------------------------------------------------


kernel_initial = tf.keras.initializers.TruncatedNormal(stddev=0.2, seed = 42)
bias_initial = tf.keras.initializers.Constant(value = 0)


# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# -------------------------------------------------------------------------------------------- 


class StochasticDepthResidual(layers.Layer):


    def __init__(self, rate = 0.2, **kwargs):

        super().__init__(**kwargs)
        self.rate = rate
        self.survival_probability = 1.0 - self.rate


    def call(self, x, training=None):

        if len(x) != 2:

            raise ValueError(f"""Input must be a list of length 2, got input with length={len(x)}.""")

        shortcut, residual = x

        b_l = keras.backend.random_bernoulli([], p = self.survival_probability)

        #return 
        residual = tf.cast(residual, dtype = tf.float32)
        shortcut = tf.cast(shortcut, dtype = tf.float32)

        return tf.cast(shortcut + b_l * residual, dtype = tf.float16) if training else tf.cast(shortcut + self.survival_probability * residual, dtype = tf.float16)

    def get_config(self):

        config = {"rate": self.rate}
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))


@keras.saving.register_keras_serializable(package = 'ResidualBlock')
class ResidualBlock(layers.Layer):
    

    def __init__(self, name, num_filters, use_APS, drop_prob = 0.2, layer_scale_init_value = 1e-6):
        
        super(ResidualBlock, self).__init__()
        
        # Parameters
        self.name_layer = name
        self.num_filters = num_filters
        self.use_APS = use_APS
        self.drop_prob = drop_prob
        self.layer_scale_init_value = layer_scale_init_value
        
        # SE blocks
        #self.cbam = CBAM(name = self.name_layer + "_CBAM", use_min = False) 

        # Feature extraction
        self.layers = tf.keras.Sequential([
            CircularPad(padding = (1, 1, 1, 1)),
            layers.Conv2D(self.num_filters, kernel_size = 7, kernel_initializer = kernel_initial, bias_initializer = bias_initial,
                            name = self.name_layer + "_conv2d_7"),
            layers.LayerNormalization(name = self.name_layer + "_layernorm"),
            
            CircularPad(padding = (1, 1, 1, 1)),
            layers.Conv2D(self.num_filters, kernel_size = 1, kernel_initializer = kernel_initial, bias_initializer = bias_initial,
                            name = self.name_layer + "_conv2d_4"),
            layers.Activation('gelu', name = self.name_layer + "_activation"),

            CircularPad(padding = (1, 1, 1, 1)),
            layers.Conv2D(self.num_filters, kernel_size = 1, kernel_initializer = kernel_initial, bias_initializer = bias_initial,
                            name = self.name_layer + "_conv2d_output")
        ], name = f"Sequential_Residual_{self.name_layer}")

        self.stochastic_depth = StochasticDepthResidual(self.drop_prob)

    def get_config(self):

        return {'name': self.name_layer, 'num_filters': self.num_filters, 'use_APS': self.use_APS,
                'drop_prob': self.drop_prob, 'layer_scale_init_value': self.layer_scale_init_value}


    def call(self, inputs):
        
        # Feature extraction inputs
        x = self.layers(inputs)

        # CBAM block
        #x = self.cbam(x)

        # Regularización
        x = self.stochastic_depth([inputs, x])
        
        return x