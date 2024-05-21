# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package = 'CircularPad')
class CircularPad(layers.Layer):
 

    def __init__(self, padding = (1, 1, 1, 1)):
 
        super(CircularPad, self).__init__()
        
        self.padding = padding


    def get_config(self):

        return {'padding': self.padding}


    def call(self, x):
 
        top_pad, bottom_pad, left_pad, right_pad = self.padding
        height_pad = tf.concat([x[:, -bottom_pad:], x, x[:, :top_pad]], axis=1)
 
        return tf.concat([height_pad[:, :, -right_pad:], height_pad, height_pad[:, :, :left_pad]], axis=2)


@keras.saving.register_keras_serializable(package = 'APSLayer')
class APSLayer(layers.Layer):


    def __init__(self, stride = 2, order = 2, name = None, **kwargs):

        super(APSLayer, self).__init__(**kwargs)

        self.name_layer = name
        self.stride = stride
        self.order = order


    def call(self, inputs):

        downsampled, max_norm_index = self.downsample(inputs)

        return downsampled, max_norm_index


    def get_config(self):

        return {'name': self.name_layer, "stride": self.stride, "order": self.order}


    @tf.function
    def downsample(self, inputs):
        
        # gather polyphase components
        polyphase_components = tf.TensorArray(tf.float32, size = 0, dynamic_size = True, clear_after_read = False)
        
        input_shape = tf.shape(inputs)
        arr_index = 0
        
        for i in range(self.stride):
        
            for j in range(self.stride):
        
                strided_matrix = tf.strided_slice(
                    inputs, 
                    begin = [0, i, j, 0], 
                    end = [0, input_shape[1], input_shape[2], input_shape[3]], 
                    strides = [1, self.stride, self.stride, 1],
                    begin_mask = 9,
                    end_mask = 9
                )
        
                strided_matrix = tf.cast(strided_matrix, dtype = tf.float32)
                polyphase_components = polyphase_components.write(arr_index, strided_matrix)
                arr_index += 1

        # find polyphase element with maximum norm
        norms = tf.map_fn(
            lambda x: tf.norm(tensor = polyphase_components.read(x), ord = self.order),
            tf.range(self.stride * 2),
            fn_output_signature=tf.float32
        )

        # Encontramos el índice del componente con la norma máxima
        max_norm_index = tf.math.argmax(norms)
        max_norm_index = tf.cast(max_norm_index, dtype = tf.int32)

        # Devolvemos el componente con la norma máxima y su índice
        return polyphase_components.read(max_norm_index), max_norm_index


@keras.saving.register_keras_serializable(package = 'APSDownsampleGivenPolyIndices')
class APSDownsampleGivenPolyIndices(layers.Layer):


    def __init__(self, stride = 2, name = None, **kwargs):

        super(APSDownsampleGivenPolyIndices, self).__init__(**kwargs)

        self.stride = stride
        self.name_layer = name


    def call(self, inputs, max_poly_indices):   

        strided_matrix = self.downsample(inputs, max_poly_indices)

        return strided_matrix


    @tf.function(jit_compile=True)
    def downsample(self, inputs, max_poly_indices):

        # Creamos una matriz de índices para 'i' y 'j'
        i, j = tf.meshgrid(tf.range(self.stride), tf.range(self.stride), indexing = 'ij')

        # Aplanamos los índices y creamos el elemento
        elem = tf.stack([tf.zeros_like(i), i, j, tf.zeros_like(i)], axis = -1)
        elem = tf.reshape(elem, [-1, 4])

        # Creamos el TensorArray 'lookup'
        lookup = tf.TensorArray(tf.int32, size = self.stride ** 2)
        lookup = lookup.unstack(elem)

        # Leemos los índices máximos de 'lookup'
        max_poly_indices = lookup.read(max_poly_indices)

        # Obtenemos la forma de las entradas
        input_shape = tf.shape(inputs)

        # Creamos la matriz strided
        strided_matrix = tf.strided_slice(
            inputs,
            begin=max_poly_indices,
            end=[0, input_shape[1], input_shape[2], input_shape[3]],
            strides=[1, self.stride, self.stride, 1],
            begin_mask=9,
            end_mask=9
        )

        # Devolvemos la matriz strided como un tensor de tipo float32
        return tf.cast(strided_matrix, dtype = tf.float32)


    def get_config(self):

        return {'name': self.name_layer, "stride": self.stride}


@keras.saving.register_keras_serializable(package = 'APSDownsampling')
class APSDownsampling(layers.Layer):


    def __init__(self, name, filtros):
        
        super(APSDownsampling, self).__init__()

        self.filtros = filtros
        self.name_layer = name

        self.padding = CircularPad((1, 1, 1, 1))

        self.aps_layer = APSLayer(name = name + "APS_Layer")
        self.downsampling = APSDownsampleGivenPolyIndices(name = name + "APSDownsampleGivenPolyIndices_Layer")

        self.conv_y = layers.Conv2D(kernel_size = 3, strides = 1, filters = filtros, name = name + "APS_conv_y")
        self.norm_y = layers.LayerNormalization(name = name + "APS_norm_y")
        self.activation_y = layers.Activation('gelu', name = name + "APS_act_y")

        self.conv_x = layers.Conv2D(kernel_size = 3, strides = 1, filters = filtros,  name = name + "APS_conv_x")
        self.norm_x = layers.LayerNormalization(name = name + "APS_norm_x")
        self.activation_x = layers.Activation('gelu', name = name + "APS_act_x")


    def get_config(self):

        return {'name': self.name_layer, "filtros": self.filtros}


    def call(self, inputs):

        y, max_norm_index = self.aps_layer(inputs)

        y = self.norm_y(y)
        y = self.activation_y(y)
        y = self.conv_y(self.padding(y))

        x = self.downsampling(inputs, max_norm_index)
        x = self.conv_x(self.padding(x))

        x = layers.Add()([x, y])
        x = self.norm_x(x)
        x = self.activation_x(x)

        return x