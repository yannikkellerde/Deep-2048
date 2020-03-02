from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

"""class MyDense(Dense):
    def __init__(self,output_dim,**kwargs):
        super(MyDense,self).__init__(output_dim,**kwargs)
    def build(self, input_shape):
        print("build",input_shape)
        super(MyDense,self).build(input_shape)
    def call(self,inputs):
        print("call",inputs.shape,self.kernel.shape)
        return super(MyDense,self).call(inputs)"""

def mlp(input_dim,hidden_layers,hidden_size,output_dim,lr=0.001,initializer='he_uniform',
        activation="relu",output_activation="linear",optimizer=Adam, loss="mse"):
    model = Sequential()
    if hidden_layers==0:
        model.add(Dense(output_dim, input_dim=input_dim,
                        activation=output_activation,kernel_initializer=initializer))
    else:
        model.add(Dense(hidden_size, input_dim=input_dim,
                            activation=activation,kernel_initializer=initializer))
        for _ in range(hidden_layers-1):
            model.add(Dense(hidden_size, activation=activation,
                            kernel_initializer=initializer))
        model.add(Dense(output_dim, activation=output_activation,
                            kernel_initializer=initializer))
    model.compile(loss=loss, optimizer=optimizer(lr))
    return model