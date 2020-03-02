from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras import backend as K

class Row_conv_Layer(Layer):
    def __init__(self, out_depth, kernel_initializer='glorot_uniform',activation=None,use_bias=True,bias_initializer='zeros'):
        super(Row_conv_Layer, self).__init__()
        self.out_depth = out_depth
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
    def build(self, input_shape):
        self.depth = input_shape[1]
        self.height = input_shape[2]
        self.width = input_shape[3]
        self.kernel = self.add_weight("kernel",
                            shape=[self.depth*(self.width+self.height-1),self.part_outputs],
                            initializer=self.kernel_initializer)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.out_depth,self.height,self.width),
                                        initializer=self.bias_initializer,
                                        name='bias')
    def call(self,input):
        