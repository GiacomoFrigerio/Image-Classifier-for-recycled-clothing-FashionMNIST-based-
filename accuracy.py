import numpy as np
import matplotlib.pyplot as plt

### EXTRACT DATA
Xtrain, Ytrain = np.load("train.npz").values()
Xtest, Ytest = np.load("test.npz").values()
Xtrain = Xtrain.reshape(Xtrain.shape[0], -1)
Xtest = Xtest.reshape(Xtest.shape[0], -1)

###EXTRACT WEIGHTS AND BIAS
data = np.load("Mult5000.npz")
W = data["arr_0"]
b = data["arr_1"]

###DEFINE FUNCTIONS
def multinomial_logreg_inference(X, W, b):
    logits = X @ W + b.T 
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return probs
def one_hot(vector, n_classes):
  return np.squeeze(np.eye(n_classes)[vector.reshape(-1)])


###PREDICTIONS CALCULUS
#TRAINING
predictions = multinomial_logreg_inference(Xtrain, W ,b )
print(predictions.shape)
predictions = np.argmax(predictions, axis = 1)
print(predictions.shape)
#TEST
Tpredictions = multinomial_logreg_inference(Xtest, W ,b )
print(Tpredictions.shape)
Tpredictions = np.argmax(Tpredictions, axis=1)
print(Tpredictions.shape)

###ACCURACY 
accuracy = (predictions == Ytrain).mean()    
Taccuracy = (Tpredictions == Ytest).mean()

#print(Ytrain[1000])
#print(predictions[1000])
print("Training accuracy:", accuracy * 100)
print("Test accuracy:", Taccuracy * 100)



""" CONFUSION MATRIX (lesson method)"""

"""We build a confusion matrix (10 by 10 in our case because now we are using 10 classes) where we count how often an audioclip of class i gets classified by class j instead."""

nclasses = 10
words = ["Tshirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","AnkleBoot"]

ConfMatrix = np.zeros((nclasses,nclasses))

for i in range(Xtest.shape[0]):
  a = Ytest[i]
  b = Tpredictions[i]
  ConfMatrix[a,b] += 1

  
Confmatrix = ConfMatrix/(Xtest.shape[0])*100
##print(Confmatrix)


#we normalize to the number of each element in the row to better watch percentages

total = ConfMatrix.sum(1, keepdims=True)
ConfMatrix = ConfMatrix/total*100

values = []
for i in range(nclasses):
 values.append(ConfMatrix[i,i])

print(values)

print("The most correct identification is related to:", max(values), "which corresponds to class", values.index(max(values)) )
print("The least correct identification is related to:", min(values), "which corresponds to class", values.index(min(values)))


##plt.matshow(ConfMatrix)
##plt.show()

plt.figure(figsize = (12,12))
plt.imshow(ConfMatrix)
for i in range(nclasses):
  for j in range(nclasses):
    plt.text(j,i, int(ConfMatrix[i,j]), color="pink")

plt.xticks(range(nclasses), words[:nclasses], rotation="vertical")
plt.yticks(range(nclasses), words[:nclasses])
plt.savefig("ConfMatrix(mlp,multlogreg)")
plt.show()



""" One hot vectors modality """

"""
Ytrain = one_hot(Ytrain, 10)
print(Ytrain.shape)
Ytest =one_hot(Ytest, 10)
#we now transform the biggest element in the array (best prediction) in 1
#and the other in zero
for i in range(predictions.shape[0]):
    Onehotprediction = np.zeros(len(predictions[i].ravel()))
    Onehotprediction[np.argmax(predictions[i])] = 1
    Onehotprediction = Onehotprediction.reshape(predictions[i].shape)
    predictions[i] = Onehotprediction
"""
""" Alternative Modality"""

"""
#we now transform the biggest element in the array (best prediction) in 1
#and the other in zero
for i in range(Tpredictions.shape[0]):
    Onehotprediction = np.zeros(len(Tpredictions[i].ravel()))
    Onehotprediction[np.argmax(Tpredictions[i])] = 1
    Onehotprediction = Onehotprediction.reshape(Tpredictions[i].shape)
    Tpredictions[i] = Onehotprediction

"""




