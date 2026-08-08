import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Test the AttentionLayer implementation
class AttentionLayer(layers.Layer):
    """Mecanismo de Attention para detectar números clave."""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_u',
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        uit = tf.tanh(tf.linalg.matmul(x, self.W) + self.b)
        ait = tf.linalg.matmul(uit, tf.expand_dims(self.u, -1))
        ait = tf.squeeze(ait, -1)
        ait = tf.nn.softmax(ait, axis=-1)
        ait = tf.expand_dims(ait, -1)
        weighted = x * ait
        output = tf.reduce_sum(weighted, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Test the layer
print("Testing AttentionLayer...")
try:
    # Create a simple test
    attention_layer = AttentionLayer()
    # Build with input shape (batch_size, sequence_length, features)
    attention_layer.build((None, 10, 8))
    # Create test input
    test_input = tf.random.normal((32, 10, 8))
    # Call the layer
    output = attention_layer(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print("AttentionLayer test PASSED!")
except Exception as e:
    print(f"AttentionLayer test FAILED: {e}")
    import traceback
    traceback.print_exc()