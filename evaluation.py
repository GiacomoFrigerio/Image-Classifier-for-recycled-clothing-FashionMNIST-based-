import numpy as np
import matplotlib.pyplot as plt
import pvml

def load_reshape(path):
    X, Y = np.load(path).values()
    X = X.reshape(X.shape[0], -1)
    return X, Y

Xtrain, Ytrain = load_reshape("train.npz")
print("Training set after reshape: ", Xtrain.shape, Ytrain.shape)
Xtest, Ytest = load_reshape("test.npz")
print("Test set after reshape: ", Xtest.shape, Ytest.shape)

nclasses = 10
words = ["Tshirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","AnkleBoot"]
##"""We want to implement normalization to improve performance"""
##""" Mean Variance Method"""
##def meanvar_normalization(Xtrain, Xval, Xtest):
##  Means = Xtrain.mean(0,keepdims=True)
##  StDev = Xtrain.std(0,keepdims=True)
##  Xtrain = (Xtrain - Means) / StDev
##  Xval = (Xval - Means) / StDev
##  Xtrain = (Xtrain - Means) / StDev
##  return Xtrain , Xval, Xtest
##
##
##""" Max Min Method (as in lecture notes)"""
##def minmax_normalization(Xtrain, Xval, Xtest):
##  """ Enters Xtrain, Xval, Xtest gives normalized version"""
##  xmin = Xtrain.min(0)
##  xmax = Xtrain.max(0)
##  Xtrain = (Xtrain - xmin) / (xmax - xmin)
##  Xval = (Xval - xmin) / (xmax - xmin)
##  Xtest = (Xtest - xmin) / (xmax - xmin)
##  return Xtrain, Xval, Xtest
##
##""" Max abs scaling (as in lecture notes)"""
##def maxabs_normalization(Xtrain, Xval, Xtest):
## amax = np.abs(Xtrain).max(0)
## Xtrain = Xtrain / amax
## Xval = Xval / amax
## Xtest = Xtest / amax
## return Xtrain, Xval, Xtest
##
##""" Whitening (as in lecture notes) """
##def whitening(Xtrain, Xval, Xtest):
## mu = Xtrain.mean(0)
## sigma = np.cov(Xtrain.T)
## evals, evecs = np.linalg.eigh(sigma)
## w = evecs / np.sqrt(evals)
## Xtrain = (Xtrain - mu) @ w
## Xval = (Xval - mu) @ w
## Xtest = (Xtest - mu) @ w
## return Xtrain, Xval, Xtest
##
""" L2 (as in lecture notes) """
def l2_normalization(X):
  q = np.sqrt((X ** 2).sum(1, keepdims=True))
  q = np.maximum(q, 1e-15) #1e-15 avoids division by zero
  X = X/q
  return X

##""" L1 (as in lecture notes) """
##def l1_normalization(X):
##  q = np.abs(X).sum(1, keepdims=True)
##  q = np.maximum(q, 1e-15)
##  X = X/q
##  return X
##


#Xtrain, Xtest = whitening(Xtrain, Xtest)
Xtrain = l2_normalization(Xtrain)
Xtest = l2_normalization(Xtest)


""" PCA """

#Number of principal components
##k = 529
##print("Number of principal components:", k)
##
##"""PCA from notes """
##def pca(Xtrain, Xtest, mincomponents=1, retvar=0.95):
##    # Compute the moments
##    mu = Xtrain.mean(0)
##    sigma = np.cov(Xtrain.T)
##    # Compute and sort the eigenvalues
##    evals, evecs = np.linalg.eigh(sigma)
##    order = np.argsort(-evals)
##    evals = evals[order]
##    # Determine the components to retain
##    r = np.cumsum(evals) / evals.sum()
##    k = 1 + (r >= retvar).nonzero()[0][0]
##    k = max(k, mincomponents)
##    w = evecs[:, order[:k]]
##    # Transform the data
##    Xtrain = (Xtrain- mu) @ w
##    Xtest = (Xtest- mu) @ w
##    return Xtrain, Xtest
##
##Xtrain, Xtest = pca(Xtrain, Xtest, k, 0.99)

""" import training model and evaluate"""
net = pvml.MLP.load("mlp(nohidden,L2).npz")

predictions, probs = net.inference(Xtrain)
acc = 100 * (predictions == Ytrain).mean()
print("Training accuracy: ", acc)


predictions, probs = net.inference(Xtest)
acc = 100 * (predictions == Ytest).mean()
print("Test accuracy: ", acc)


""" CONFUSION MATRIX (lesson method)"""

"""We build a confusion matrix (10 by 10 in our case because now we are using 10 classes) where we count how often an audioclip of class i gets classified by class j instead."""

##ConfMatrix = np.zeros((nclasses,nclasses))
##
##for i in range(Xtest.shape[0]):
##  a = Ytest[i]
##  b = predictions[i]
##  ConfMatrix[a,b] += 1
##
##  
##Confmatrix = ConfMatrix/(Xtest.shape[0])*100
####print(Confmatrix)
##
##
###we normalize to the number of each element in the row to better watch percentages
##
##total = ConfMatrix.sum(1, keepdims=True)
##ConfMatrix = ConfMatrix/total*100
##
##values = []
##for i in range(nclasses):
## values.append(ConfMatrix[i,i])
##
##print(values)
##
##print("The most correct identification is related to:", max(values), "which corresponds to class", values.index(max(values)) )
##print("The least correct identification is related to:", min(values), "which corresponds to class", values.index(min(values)))
##
##
####plt.matshow(ConfMatrix)
####plt.show()
##
##plt.figure(figsize = (12,12))
##plt.imshow(ConfMatrix)
##for i in range(nclasses):
##  for j in range(nclasses):
##    plt.text(j,i, int(ConfMatrix[i,j]), color="pink")
##
##plt.xticks(range(nclasses), words[:nclasses], rotation="vertical")
##plt.yticks(range(nclasses), words[:nclasses])
##plt.savefig("ConfMatrix(mlp,3hidden)")
##plt.show()

""" Show weights (without hidden layers)"""

w = net.weights[0]
print(w.shape)
fig = plt.figure(figsize=(12,12))
columns = 5
rows = 2
for i in range(1,nclasses+1):
  fig.add_subplot(rows,columns,i)
  plt.imshow(w[:, i-1].reshape(28,28), cmap="plasma", vmin=-0.6, vmax=0.6, aspect = "auto")
  plt.axis("off")
plt.savefig("Weights(mlp,nohidden, L2)")
plt.show()
  
