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

""" PCA """

###Number of principal components
k = 729
print("Number of principal components:", k)

"""PCA from notes """
def pca(Xtrain, Xtest, mincomponents=1, retvar=0.95):
    # Compute the moments
    mu = Xtrain.mean(0)
    sigma = np.cov(Xtrain.T)
    # Compute and sort the eigenvalues
    evals, evecs = np.linalg.eigh(sigma)
    order = np.argsort(-evals)
    evals = evals[order]
    # Determine the components to retain
    r = np.cumsum(evals) / evals.sum()
    k = 1 + (r >= retvar).nonzero()[0][0]
    k = max(k, mincomponents)
    w = evecs[:, order[:k]]
    # Transform the data
    Xtrain = (Xtrain- mu) @ w
    Xtest = (Xtest- mu) @ w
    return Xtrain, Xtest

Xtrain, Xtest = pca(Xtrain, Xtest, k, 0.99)

""" import training model and evaluate"""
net = pvml.MLP.load("mlp(nohidden,PCA).npz")

predictions, probs = net.inference(Xtrain)
acc = 100 * (predictions == Ytrain).mean()
print("Training accuracy: ", acc)


predictions, probs = net.inference(Xtest)
acc = 100 * (predictions == Ytest).mean()
print("Test accuracy: ", acc)


""" Show weights (without hidden layers)"""

w = net.weights[0]
print(w.shape)
fig = plt.figure(figsize=(12,12))
columns = 5
rows = 2
for i in range(1,nclasses+1):
  fig.add_subplot(rows,columns,i)
  plt.imshow(w[:, i-1].reshape(27,27), cmap="plasma", vmin=-0.6, vmax=0.6, aspect = "auto")
  plt.axis("off")
plt.savefig("Weights(mlp,nohidden, PCA)")
plt.show()
  

