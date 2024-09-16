import numpy as np

traindata = input("Input train data name \n")
print("File inserted" ,traindata)
testdata = input("Input test data name \n")
print("File inserted" ,testdata)
model = input("Input model name \n")
print("Model inserted" ,model)

##Xtrain, Ytrain = np.load("train.npz").values()
##Xtest, Ytest = np.load("test.npz").values()

data = np.load(traindata)
Xtrain = data["arr_0"]
Ytrain = data["arr_1"]

data = np.load(testdata)
Xtest = data["arr_0"]
Ytest = data["arr_1"]

Xtrain = Xtrain.reshape(Xtrain.shape[0], -1)
Xtest = Xtest.reshape(Xtest.shape[0], -1)

data = np.load(model)
W = data["arr_0"]
b = data["arr_1"]

def multinomial_logreg_inference(X, W, b):
    logits = X @ W + b.T 
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return probs


def one_hot(vector, n_classes):
  return np.squeeze(np.eye(n_classes)[vector.reshape(-1)])

Ytrain = one_hot(Ytrain, 10)
print(Ytrain.shape)
Ytest =one_hot(Ytest, 10)


predictions = multinomial_logreg_inference(Xtrain, W ,b )
#we now transform the biggest element in the array (best prediction) in 1
#and the other in zero
for i in range(predictions.shape[0]):
    Onehotprediction = np.zeros(len(predictions[i].ravel()))
    Onehotprediction[np.argmax(predictions[i])] = 1
    Onehotprediction = Onehotprediction.reshape(predictions[i].shape)
    predictions[i] = Onehotprediction
    
accuracy = (predictions == Ytrain).mean()

Tpredictions = multinomial_logreg_inference(Xtest, W ,b )
#we now transform the biggest element in the array (best prediction) in 1
#and the other in zero
for i in range(Tpredictions.shape[0]):
    Onehotprediction = np.zeros(len(Tpredictions[i].ravel()))
    Onehotprediction[np.argmax(Tpredictions[i])] = 1
    Onehotprediction = Onehotprediction.reshape(Tpredictions[i].shape)
    Tpredictions[i] = Onehotprediction
    
Taccuracy = (Tpredictions == Ytest).mean()

#print(Ytrain[1000])
#print(predictions[1000])
print("Training accuracy:", accuracy * 100)
print("Test accuracy:", Taccuracy * 100)
