from tensorflow.keras.layers import Layer,Flatten,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras import backend as K
import tensorflow as tf

class PartsLayer(Layer):
    def __init__(self, part_outputs, kernel_initializer='glorot_uniform',activation=None,use_bias=True,bias_initializer='zeros'):
        super(PartsLayer, self).__init__()
        self.part_outputs = part_outputs
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.parts = []
        self.part_length = input_shape[2]
        for i in range(input_shape[1]):
            self.parts.append(self.add_weight("part"+str(i),
                              shape=[self.part_length,self.part_outputs],
                              initializer=self.kernel_initializer))
        if self.use_bias:
            self.bias = self.add_weight(shape=(input_shape[1],self.part_outputs),
                                        initializer=self.bias_initializer,
                                        name='bias')
        else:
            self.bias = None
        self.built = True
    def call(self, input):
        begin = 0
        out_slices = []
        for i in range(len(self.parts)):
            part = self.parts[i]
            put = input[:,i]
            res = K.dot(put, part)
            out_slices.append(res)
            begin+=self.part_length
        output = tf.stack(out_slices)
        output = tf.transpose(output,perm=[1,0,2])
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
    
def build_smart_nn(hidden_part_size,part_layers,hidden_layers,
                   hidden_size,output_dim,lr=0.001,initializer='glorot_uniform',activation="relu",
                   output_activation="linear",optimizer=Adam, loss="mse"):
    model = Sequential()
    model.add(PartsLayer(hidden_part_size,activation=activation,kernel_initializer=initializer))
    for parts in range(part_layers-1):
        model.add(PartsLayer(hidden_part_size,activation=activation,kernel_initializer=initializer))
    model.add(Flatten())
    for _ in range(hidden_layers):
        model.add(Dense(hidden_size, activation=activation,
                        kernel_initializer=initializer))
    model.add(Dense(output_dim, activation=output_activation,
                        kernel_initializer=initializer))
    model.compile(loss=loss, optimizer=optimizer(lr))
    return model