# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from ..Layers.APS import CircularPad, APSDownsampling
from ..Layers.ResidualBlock import ResidualBlock


# --------------------------------------------------------------------------------------------
# SETS THE SEED OF THE RANDOM NUMBER GENERATOR
# --------------------------------------------------------------------------------------------


kernel_initial = tf.keras.initializers.TruncatedNormal(stddev = 0.2, seed = 42)
bias_initial = tf.keras.initializers.Constant(value = 0)


# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------------------------------------------------   


def mcnn_classifier(input_shape, num_classes, residual_copies = [2, 2, 2, 2, 2], num_layers = 5, filter_sizes = [32, 32, 64, 64, 64], kernel_size = [3, 3, 3, 3, 3]):

    # Input data
    inputs = layers.Input(input_shape, name = "Input")

    # Copia de la entrada
    x = inputs
    
    for lyr in range(num_layers):

        if lyr == 0:

            x = tf.keras.Sequential([
                CircularPad(padding = (1,1,1,1)),
                layers.Conv2D(filters = filter_sizes[lyr], kernel_size = kernel_size[lyr], use_bias = bias_initial, kernel_initializer = kernel_initial),
                layers.LayerNormalization(),
                layers.Activation('gelu'),

                CircularPad(padding = (1,1,1,1)),
                layers.Conv2D(filters = filter_sizes[lyr], kernel_size = kernel_size[lyr],use_bias = bias_initial, kernel_initializer = kernel_initial),
                layers.LayerNormalization(),
                layers.Activation('gelu'),
            ], name = "Initial")(x)
                
        else:

            # Capas residuales
            for i in range(residual_copies[lyr - 1]):
                
                x = ResidualBlock(name = f"Residual_Block_{lyr}_{i}", num_filters = filter_sizes[lyr], use_APS = True)(x)

            # Obtencion de los filtros dependiendo de la capa en la que nos encontremos
            filtros = filter_sizes[lyr + 1] if lyr + 1 < num_layers else 64

            # Donwnsampling
            x = APSDownsampling(name = f"APSDownsampling_{lyr}", filtros = filtros)(x)
            
    # SPP Block
    x = layers.GlobalAveragePooling2D()(x)

    # Softmax
    outputs = layers.Dense(num_classes, activation = "softmax")(x)

    return keras.Model(inputs = inputs, outputs = outputs, name = "MCNN")