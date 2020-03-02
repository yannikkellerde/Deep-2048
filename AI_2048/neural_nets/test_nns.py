from AI_2048.env.game import Game_2048
from AI_2048.neural_nets.parts_based import build_smart_nn
from AI_2048.neural_nets.mlp import mlp
import random
import numpy as np
game = Game_2048()
def xor(a, b):
    return (a or b) and (not (a and b))
def spit_out_test_data():
    input = game.convert_to_nn_input_2D(game.space_2d.sample())
    #input = np.array([[random.randint(0,1) for _ in range(16)] for _ in range(12)])
    output = np.zeros((len(input),))
    for i in range(len(input)):
        put = input[i]
        if xor(sum(put[:4])>sum(put[4:8]),sum(put[2:6])>sum(put[6:10])):
            output[i]=1
        if xor(sum(put[7:12])>sum(put[0:6]), put[2]):
            output[i]=0
        if xor(sum(put[8:12])>sum(put[3:7]),sum(put[1:3])>sum(put[9:11])):
            output[i]=1
    for i in range(len(output)):
        if i==len(output)-1:
            output[i]=xor(output[i],output[0])
        else:
            output[i]=xor(output[i],output[i+1])
    return input,output
def get_batch(size):
    batch_in = []
    batch_flat = []
    targets = []
    for _ in range(size):
        input,output = spit_out_test_data()
        batch_flat.append(input.flatten())
        batch_in.append(input)
        targets.append(output)
    return np.array(batch_flat),np.array(batch_in,dtype=np.float32),np.array(targets)

smart = build_smart_nn(20,1,2,27,game.space_2d.nvec[0],output_activation="sigmoid",lr=0.01)
normal = mlp(game.space_1d.n,3,27,game.max_power,output_activation="sigmoid",lr=0.01)
for i in range(1):
    batch_flat,batch_in,target = get_batch(256)
    smart_loss = smart.train_on_batch(batch_in, target)
    normal_loss = normal.train_on_batch(batch_flat,target)
    if i%10==0:
        print(f"Iteration {i}, smart_loss {smart_loss}, normal_loss {normal_loss}")
print(smart.summary())
print(normal.summary())
"""def spit_xor_batch(size):
    inputs=np.array([[random.randint(0,1) for _ in range(2)] for _ in range(size)])
    outputs = np.zeros(size)
    for i in range(len(inputs)):
        if (inputs[i][0] or inputs[i][1]) and (not (inputs[i][0] and inputs[i][1])):
            outputs[i]=1
    return inputs,outputs
model = mlp(2,0,0,1,output_activation="sigmoid",lr=0.01)
for i in range(100000):
    batch_in,target = spit_xor_batch(1024)
    loss = model.train_on_batch(batch_in, target)
    if i%10==0:
        print(f"Iteration {i}, loss {loss}")"""