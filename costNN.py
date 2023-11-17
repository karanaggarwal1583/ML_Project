
import numpy as np
import neurolab as nl
import time


def costNN(x,inputs,outputs,net):
    trainInput=inputs
    
    
    trainOutput=outputs
    
    numInputs=np.shape(trainInput)[1] #number of inputs
    
    
    HiddenNeurons = (net.layers[0].np['b'][:].shape[0])
    ######################################

    
    split1=int((HiddenNeurons)*numInputs)
    split2 =int(split1+(HiddenNeurons)*(HiddenNeurons))
    split3=  int(split2+HiddenNeurons)
    split4=  int(split3+HiddenNeurons)
    split5 = int(split4+HiddenNeurons)
 
    # input_w = 3X8 (HiddenNeurons*numInputs) 
    input_w =x[0:split1].reshape(HiddenNeurons,numInputs)
                       
    # layer_w = 1 X 3 (HiddenNeurons)
    layer_w=x[split1:split2].reshape(HiddenNeurons,HiddenNeurons)
    layer2_w=x[split2:split3].reshape(1,HiddenNeurons)
 
    # input_bias = hiddenNeurons
    input_bias=x[split3:split4].reshape(1,HiddenNeurons)
    #input_bias = np.array([0.4747,-1.2475,-1.2470])
    layer_bias=x[split4:split5].reshape(1,HiddenNeurons)
    # bias_2 = 1
    bias_2 =x[split5:split5+1]

    
    net.layers[0].np['w'][:] = input_w
    net.layers[1].np['w'][:] = layer_w
    net.layers[2].np['w'][:] = layer2_w
    net.layers[0].np['b'][:] = input_bias
    net.layers[1].np['b'][:] = layer_bias
    net.layers[2].np['b'][:] = bias_2
    
    
    pred=net.sim(trainInput).reshape(len(trainOutput))
    
    mse = ((pred - trainOutput) ** 2).mean(axis=None)
    
    
    
    return mse