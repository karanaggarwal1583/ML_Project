

from sklearn.metrics import confusion_matrix
#from sklearn.metrics import recall_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import neurolab as nl
import time

def evaluateNetClassifier(solution,inputs,outputs,net):
    
    
    x=solution.bestIndividual
    
    printAcc=[]
    
    trainInput=inputs
    trainOutput=outputs
    
    numInputs=np.shape(trainInput)[1] #number of inputs
    
    
    #number of hidden neurons
    HiddenNeurons = (net.layers[0].np['b'][:].shape[0])
    
    ######################################

    
    split1=int((HiddenNeurons)*numInputs)
    split2 =int(split1+(HiddenNeurons)*(HiddenNeurons))
    split3=  int(split2+HiddenNeurons)
    split4=  int(split3+HiddenNeurons)
    split5 = int(split4+HiddenNeurons)
    HiddenNeurons =int(HiddenNeurons)
    
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
    pred_act = []
    for i in pred:
        if i < -0.50:
            pred_act.append(-1)
        elif i >0.50:
            pred_act.append(1)
        else:
            pred_act.append(0)
            
    
    # pred=np.round(pred).astype(int)   
    # trainOutput=trainOutput.astype(int) 
    # pred=np.clip(pred, 0, 1)

    ConfMatrix=confusion_matrix(trainOutput, pred_act)
    

    #print(ConfMatrix)
    #print(trainOutput)
    #print(pred)
    #time.sleep(5)
    ConfMatrix1D=ConfMatrix.flatten()
    #print(ConfMatrix1D)
    print(accuracy_score(trainOutput, pred_act,normalize=True))
    printAcc.append(accuracy_score(trainOutput, pred_act,normalize=True)) 
    
    classification_results= np.concatenate((printAcc,ConfMatrix1D))
    #print(classification_results)
    return classification_results
    
    
    
    
