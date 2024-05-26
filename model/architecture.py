# %% Libraries

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ConvNeXtSmall
import numpy as np
from .layers.APS import CircularPad, APSDownsampling
from .layers.SE import SqueezeAndExcitation
import math
import keras_cv
import config as cg

# %% Functions

def data_augmentation_layer(inputs: tf.Tensor, seed: int = 42) -> tf.Tensor:

    """
    Apply data augmentation to the input data.

    Args:
        inputs (tf.Tensor): The input data.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        tf.Tensor: The augmented input data.
    """
    
    tf.random.set_seed(seed)

    x = inputs

    if tf.random.uniform([]) < 0.5:

        x = keras_cv.layers.GridMask(ratio_factor = (0, 0.5), rotation_factor = 0, fill_mode = "constant", fill_value = 0.0, seed = seed)(x)

    x = layers.RandomCrop(height = 128, width = 128, seed = seed)(x)

    if tf.random.uniform([]) < 0.5:
        
        x = layers.GaussianNoise(stddev = 0.1, seed = seed)(x)

    if tf.random.uniform([]) < 0.5:
        
        x = layers.RandomFlip("horizontal", seed = seed)(x)

    return x

# %% ConvNeXtSmall pretrained model

def build_model_pretrained(input_shape: tuple, num_classes: int, fc_layers: list = [256, 128], dropout: float = 0.2) -> keras.Model:
    
    """
    Build a classification model using a pre-trained ConvNeXtSmall model.

    Args:
        input_shape (tuple): The shape of the input data.
        num_classes (int): The number of classes in the classification problem.
        fc_layers (list): A list of integers specifying the number of neurons in each fully connected layer. Defaults to [256, 128].
        dropout (float): The dropout rate for the fully connected layers. Defaults to 0.2.

    Returns:
        keras.Model: The built classification model.
    """

    # Load the pre-trained ConvNeXtSmall model
    base_model = ConvNeXtSmall(weights = "imagenet", include_top = False, input_shape = input_shape)
    
    # Freeze the layers of the pre-trained model due to limited memory
    for layer in base_model.layers:

        # Set the trainable attribute of each layer to False
        layer.trainable = False

    # Create an input layer to introduce the data
    inputs = layers.Input(shape = input_shape)

    # Use data augmentation
    x = data_augmentation_layer(inputs)
    
    # Use the features obtained from the pre-trained model
    x = base_model(x)
    
    # Reduce the dimensionality using global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Use a few fully connected layers (MLPs)
    for fc in fc_layers:

        # Add a dense layer with the specified number of neurons and Gelu activation
        x = layers.Dense(fc, activation='gelu')(x)

        # Add a dropout layer with the specified dropout rate
        x = layers.Dropout(dropout)(x)
    
    # Get the output of the classification using a softmax activation function
    outputs = layers.Dense(num_classes, activation = "softmax")(x)
    
    # Create the model
    return tf.keras.Model(inputs = inputs, outputs = outputs, name = "MusicGenreClassifier")

# %% CCT

class CCTTokenizer(layers.Layer): 

    """
    This class defines the CCTTokenizer layer used in the model.
    """
    
    def __init__(self, kernel_size: int = 3, num_conv_layers: int = 3, num_output_channels: list[int] = [32, 64, cg.projection_dim], **kwargs):
        
        """
        Initialize the CCTTokenizer layer with the given parameters.

        Args:
            kernel_size (int, optional): The size of the kernel used in the convolutional layers. Defaults to 3.
            num_conv_layers (int, optional): The number of convolutional layers in the model. Defaults to 3.
            num_output_channels (list, optional): The number of output channels for each convolutional layer. Defaults to [32, 64, projection_dim].
        """

        super().__init__(**kwargs)
        self.conv_model = self._build_conv_model(kernel_size, num_conv_layers, num_output_channels)

    def _build_conv_model(self, kernel_size, num_conv_layers, num_output_channels):

        model = keras.Sequential()

        for i in range(num_conv_layers):

            layers_to_add = [
                CircularPad(padding = (1, 1, 1, 1)),
                layers.LayerNormalization(),
                layers.Activation('gelu'),
                SqueezeAndExcitation(name = f"SE_{i}", num_filters = num_output_channels[i])
            ]

            if i % 2 == 0:

                layers_to_add.insert(1, layers.Conv2D(filters = num_output_channels[i], kernel_size = kernel_size, use_bias = False, kernel_initializer = "he_normal"))
            
            else:

                layers_to_add.insert(1, layers.SeparableConv2D(filters = num_output_channels[i], kernel_size = kernel_size, use_bias = False, kernel_initializer = "he_normal"))

            model.add(keras.Sequential(layers_to_add))
            model.add(APSDownsampling(name = f"APSDownsampling_{i}", filters = num_output_channels[i]))
        
        return model

    def call(self, images: tf.Tensor) -> tf.Tensor:

        outputs = self.conv_model(images)
        return tf.reshape(outputs, (-1, tf.shape(outputs)[1] * tf.shape(outputs)[2], tf.shape(outputs)[-1]))

class SequencePooling(layers.Layer):

    """
    This class defines the SequencePooling layer used in the model. It inherits from the keras Layer class.
    The SequencePooling layer applies attention mechanism to pool the sequence of vectors into a single vector representation.
    """
    
    def __init__(self):

        super(SequencePooling, self).__init__()
    
        self.attention = layers.Dense(1) 

    def call(self, x: tf.Tensor) -> tf.Tensor:

        """
        Forward pass of the SequencePooling layer.

        Args:
            x (Tensor): The input tensor of shape (batch_size, sequence_length, num_features).

        Returns:
            Tensor: The output tensor of shape (batch_size, num_features).
        """

        # Compute the attention weights for each sequence element
        attention_weights = tf.nn.softmax(self.attention(x), axis=1)
        
        # Transpose the attention weights to match the shape of the input tensor
        # Compute the weighted sum of the input tensor along the sequence dimension
        weighted_representation = tf.matmul(tf.transpose(attention_weights, perm=(0, 2, 1)), x) 

        # Remove the singleton dimension
        return tf.squeeze(weighted_representation, -2)

def mlp(x: tf.Tensor, hidden_units: list[int], dropout_rate: float) -> tf.Tensor:

    """
    This function defines a multi-layer perceptron (MLP) with GELU activation and dropout.

    Args:
        x (tf.Tensor): The input tensor.
        hidden_units (list[int]): The number of units in each hidden layer.
        dropout_rate (float): The dropout rate.

    Returns:
        tf.Tensor: The output tensor of the MLP.
    """

    for units in hidden_units:

        x = layers.Dense(units, activation = 'gelu')(x)
        x = layers.Dropout(dropout_rate)(x)
    
    return x

class StochasticDepth(layers.Layer):

    """
    This class defines the StochasticDepth layer used in the model. It inherits from the keras Layer class.
    The StochasticDepth layer randomly drops out entire layers during training.
    """
    
    def __init__(self, drop_prop: float, **kwargs):

        """
        Initialize the StochasticDepth layer with the given dropout probability.

        Args:
            drop_prop (float): The dropout probability.
        """

        super(StochasticDepth, self).__init__(**kwargs)
    
        self.drop_prob = drop_prop 
        self.seed_generator = tf.keras.utils.set_random_seed(1337) 

    def call(self, x: tf.Tensor, training: bool = None) -> tf.Tensor:

        """
        Forward pass of the StochasticDepth layer.

        Args:
            x (Tensor): The input tensor.
            training (bool, optional): Whether the layer is in training mode. Defaults to None.

        Returns:
            Tensor: The output tensor of the StochasticDepth layer.
        """

        # Only apply stochastic depth if the layer is in training mode
        if training:

            # Compute the keep probability
            keep_prob = 1 - self.drop_prob

            # Compute the shape of the random tensor
            shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)

            # Generate a random tensor
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1, seed = self.seed_generator)
            
            # Apply the floor function to the random tensor to create a binary mask
            random_tensor = tf.floor(random_tensor)

            # Cast the tensors to the correct data types
            x = tf.cast(x, dtype=tf.float32)

            # Apply the binary mask to the input tensor and scale the result
            return tf.cast((x / keep_prob) * tf.cast(random_tensor, dtype=tf.float32), dtype=tf.float16)

        return x

class MultiHeadAttentionLSA(layers.MultiHeadAttention):

    """
    This class defines a variant of the MultiHeadAttention layer that includes a learnable scaling factor (tau).
    """

    def __init__(self, **kwargs):

        super(MultiHeadAttentionLSA, self).__init__(**kwargs)

        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable = True)

    def _compute_attention(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, attention_mask: bool = None, training: bool = None) -> tuple:

        """
        Compute the attention scores.

        Args:
            query (Tensor): The query tensor.
            key (Tensor): The key tensor.
            value (Tensor): The value tensor.
            attention_mask (Tensor, optional): A mask tensor to apply to the attention scores. Defaults to None.
            training (bool, optional): Whether the layer is in training mode. Defaults to None.

        Returns:
            Tensor: The output tensor of the attention computation.
            Tensor: The attention scores.
        """

        query = tf.cast(tf.multiply(tf.cast(query, dtype = tf.float32), 1.0 / self.tau), dtype = tf.float32)
        key = tf.cast(key, dtype = tf.float32)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        
        # Create the attention mask with the same size as attention_scores
        seq_len = tf.shape(attention_scores)[-1]
        attention_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(attention_scores, training = training)
        attention_output = tf.einsum(self._combine_equation, attention_scores_dropout, value)

        return attention_output, attention_scores

class TokenLearnerModule(tf.keras.Model):

    """
    This class defines a TokenLearnerModule which is a subclass of tf.keras.Model.
    It learns to select tokens from the input tensor and applies a multi-layer perceptron (MLP) to it.
    """

    def __init__(self, num_tokens: int = 4, bottleneck_dim: int = 64, dropout_rate: float = 0.2):
        
        """
        Initializes the TokenLearnerModule.

        Args:
            num_tokens (int): The number of tokens to be selected.
            bottleneck_dim (int, optional): The dimension of the bottleneck layer in the MLP. Defaults to 64.
            dropout_rate (float, optional): The dropout rate for the dropout layer in the MLP. Defaults to 0.0.
        """

        super(TokenLearnerModule, self).__init__()

        self.num_tokens = num_tokens
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate
        self.layernorm = layers.LayerNormalization()  
        self.mlp = self._create_mlp()  
        self.softmax = layers.Softmax(axis = -1)

    def _create_mlp(self):

        """
        Creates the multi-layer perceptron (MLP) used in the token learner module.

        Returns:
            tf.keras.Sequential: The MLP model.
        """
        
        mlp = tf.keras.Sequential([
            layers.Dense(self.bottleneck_dim, activation = 'gelu'), 
            layers.Dropout(self.dropout_rate), 
            layers.Dense(self.num_tokens)
        ])

        return mlp

    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        """
        Forward pass for the TokenLearnerModule.

        Args:
            inputs (tf.Tensor): The input tensor. Can be 3D or 4D.

        Returns:
            tf.Tensor: The output tensor after token selection and feature aggregation.
        """
        # If the input tensor is 4D, reshape it to 3D
        if inputs.ndim == 4:

            input_shape = tf.shape(inputs)
            n, h, w, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
            # Reshape to (n, h * w, c)
            inputs = tf.reshape(inputs, [n, h * w, c])  

        feature_shape = tf.shape(inputs)

        # Token selection
        selected = inputs
        selected = self.layernorm(selected) 
        selected = self.mlp(selected)  
        # Reshape to (n, h * w, num_tokens)
        selected = tf.reshape(selected, (feature_shape[0], -1, self.num_tokens))  
        # Transpose to (n, num_tokens, h * w)
        selected = tf.transpose(selected, [0, 2, 1])  
        selected = self.softmax(selected)

        # Feature aggregation
        feat = inputs
        # Reshape to (n, h*w, c)
        feat = tf.reshape(feat,(feature_shape[0], -1, feature_shape[-1]))  
        
        # Aggregate features using the selected tokens
        return tf.einsum('...si,...id->...sd', selected, feat)  


def build_model_scratch(input_shape: tuple, num_classes: int, fc_layers: list = [256, 128], dropout: float = 0.2, num_heads: int = 2, transformer_layers: int = 2,
                        stochastic_depth_rate: float = 0.1, eps: float = 1e-6) -> keras.Model:
    
    """
    This function builds a music genre classifier model from scratch using the specified parameters.

    Args:
        input_shape (tuple): The shape of the input data.
        num_classes (int): The number of classes for classification.
        fc_layers (list, optional): The number of units in each fully connected layer. Defaults to [256, 128].
        dropout (float, optional): The dropout rate. Defaults to 0.2.
        num_heads (int, optional): The number of attention heads in the multi-head attention layer. Defaults to 2.
        transformer_layers (int, optional): The number of transformer layers. Defaults to 2.
        stochastic_depth_rate (float, optional): The dropout rate for stochastic depth. Defaults to 0.1.
        eps (float, optional): A small constant for numerical stability of layer normalization. Defaults to 1e-6.

    Returns:
        keras.Model: The constructed Keras model.
    """
    
    inputs = layers.Input(input_shape)

    # Augment data.
    augmented = data_augmentation_layer(inputs)

    # Encode patches.
    cct_tokenizer = CCTTokenizer()
    encoded_patches = cct_tokenizer(augmented)

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):

        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon = eps)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = MultiHeadAttentionLSA(num_heads = num_heads, key_dim = cg.projection_dim, dropout = dropout)(x1, x1)

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon = eps)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units = cg.transformer_units, dropout_rate = dropout)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

        # Token Learner
        if i == transformer_layers // 2:

            _, hh, c = encoded_patches.shape
            h = int(math.sqrt(hh))
            # Reshape to (B, h, h, projection_dim)
            encoded_patches = layers.Reshape((h, h, c))(encoded_patches)  
            encoded_patches = TokenLearnerModule(dropout_rate = dropout)(encoded_patches)
    
    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon = eps)(encoded_patches)
    weighted_representation = SequencePooling()(representation)

    # Classify outputs.
    logits = layers.Dense(num_classes, activation = 'softmax')(weighted_representation)

    return keras.Model(inputs = inputs, outputs = logits, name = "MusicGenreClassifier")