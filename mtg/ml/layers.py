import tensorflow as tf
import numpy as np
from mtg.ml.nn import MLP

class Dense(tf.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        name = None,
        initializer = tf.initializers.GlorotNormal(),
        activation = tf.nn.relu,
        use_bias = True,
    ):
        super().__init__(name=name)

        self.activation = activation
        self.use_bias = use_bias

        self.w = tf.Variable(
            initializer([in_dim, out_dim]),
            dtype = tf.float32,
            name = self.name + '_w',
            trainable=True,
        )
        if self.use_bias:
            self.b = tf.Variable(
                tf.zeros([out_dim]),
                dtype=tf.float32,
                trainable=True,
                name=self.name + '_b',
            )

    def __call__(self, x, training=None):
        rank = x.shape.rank
        if rank == 2 or rank is None:
            y = tf.matmul(x, self.w)
        else:
            y = tf.tensordot(x, self.w, [[rank - 1], [0]])
            if not tf.executing_eagerly():
                shape = x.shape.as_list()
                output_shape = shape[:-1] + [self.w.shape[-1]]
                y.set_shape(output_shape)
        if self.use_bias:
            y = tf.nn.bias_add(y, self.b)
        if self.activation is not None:
            y = self.activation(y)
        return y

class LayerNormalization(tf.Module):
    def __init__(
        self,
        last_dim,
        epsilon=1e-6,
        center=True,
        scale=True,
        name=None,
    ):
        super().__init__(name=name)
        self.center = center
        self.epsilon = epsilon
        #current implementation can only normalize off last axis
        self.axis = -1
        if scale:
            self.gamma = tf.Variable(tf.ones(last_dim), dtype=tf.float32, trainable=True, name=self.name + "_gamma")
        else:
            self.gamma = None
        if center:
            self.beta = tf.Variable(tf.zeros(last_dim), dtype=tf.float32, trainable=True, name=self.name + "_beta")
        else:
            self.beta = None

    def __call__(self, x, training=None):
        mu, sigma = tf.nn.moments(x, self.axis, keepdims=True)
        # Compute layer normalization using the batch_normalization function.
        outputs = tf.nn.batch_normalization(
            x,
            mu,
            sigma,
            offset=self.beta,
            scale=self.gamma,
            variance_epsilon=self.epsilon
        )
        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(x.shape)
        return outputs

class MultiHeadAttention(tf.Module):
    """
    tf implementation of multi-headed attention. For the Draft bot, this will use
    self attention only. With that in mind, the parameters mean:

        d_model is the final dimension for the embedding representation post-attention

        num_heads is the number of contextual ways to look at the information
    
        k_dim will be equal to the number of time steps in a draft
            (e.g. 45), and the mask will prevent lookahead (e.g. the mask for P1P3 will look
            like [0, 0, 0, 1, 1, . . ., 1]), meaning that only information at P1P1, P1P2, and
            P1P3 can be used to make a prediction.

    https://www.tensorflow.org/text/tutorials/transformer
    """
    def __init__(self, d_model, k_dim, num_heads, v_dim=None, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        v_dim = k_dim if v_dim is None else v_dim

        self.wq = Dense(k_dim, d_model, activation=None, name=self.name + "_wq")
        self.wk = Dense(k_dim, d_model, activation=None, name=self.name + "_wk")
        self.wv = Dense(v_dim, d_model, activation=None, name=self.name + "_wv")

        self.dense = Dense(k_dim, d_model, activation=None, name=self.name + "_attout")

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    @tf.function
    def __call__(self, v, k, q, mask, training=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q, training=training)  # (batch_size, seq_len, d_model)
        k = self.wk(k, training=training)  # (batch_size, seq_len, d_model)
        v = self.wv(v, training=training)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                        (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention, training=training)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
        output, attention_weights
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            #expand mask dimension to allow for addition on all attention heads
            scaled_attention_logits += (tf.expand_dims(mask,1) * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

class Embedding(tf.Module):
    def __init__(self, num_items, emb_dim, initializer=tf.initializers.GlorotNormal(), name=None, activation=None,):
        super().__init__(name=name)
        self.embedding = tf.Variable(initializer(shape=(num_items, emb_dim)), dtype=tf.float32, name=self.name + "_embedding")
        self.activation = activation

    def __call__(self, x, training=None):
        embeddings = tf.gather(self.embedding, x)
        if self.activation is not None:
            embeddings = self.activation(embeddings)
        return embeddings

class ConcatEmbedding(tf.Module):
    """
    Lets say you want an embedding that is a concatenation of the abstract object and data about the object

    so we learn a normal one hot embedding, and then have an MLP process the data about the object and concatenate the two.
    """
    def __init__(
        self,
        num_items,
        emb_dim,
        item_data,
        dropout=0.0,
        n_h_layers=1,
        initializer=tf.initializers.GlorotNormal(),
        name=None,
        activation=None,
        start_act=None,
        middle_act=None,
        out_act=None,
    ):
        super().__init__(name=name)
        assert item_data.shape[0] == num_items
        self.item_data = item_data
        self.item_MLP = MLP(
            in_dim=item_data.shape[0],
            start_dim=item_data.shape[0]//2,
            out_dim=emb_dim//2,
            n_h_layers=n_h_layers,
            dropout=dropout,
            name="item_data_mlp",
            start_act=start_act,
            middle_act=middle_act,
            out_act=out_act,
            style="bottleneck",
        )
        self.embedding = tf.Variable(initializer(shape=(num_items, emb_dim//2)), dtype=tf.float32, name=self.name + "_embedding")
        self.activation = activation

    def __call__(self, x, training=None):
        item_embeddings = tf.gather(self.embedding, x)
        item_data = tf.gather(self.item_data, x)
        data_embeddings = self.item_MLP(item_data)
        embeddings = tf.concat([item_embeddings, data_embeddings], axis=-1)
        if self.activation is not None:
            embeddings = self.activation(embeddings)
        return embeddings