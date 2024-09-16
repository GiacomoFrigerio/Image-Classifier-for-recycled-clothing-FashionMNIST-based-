import numpy as np
import matplotlib.pyplot as plt
import pvml


### compute accuracy
def accuracy(net, X, Y):
    ''' Compute the accuracy.

    : param net: MLP neural network.
    : param X: array like.
    : param Y: array like.
    : return acc * 100: number.
    
    '''
    labels, probs = net.inference(X)
    acc = (labels == Y).mean()
    return acc * 100

def load_reshape(path):
    X, Y = np.load(path).values()
    X = X.reshape(X.shape[0], -1)
    return X, Y

def l2_normalization(X):
    q = np.sqrt((X ** 2).sum(1, keepdims = True))
    q = np.maximum(q, 1e-15)
    X = X / q
    return X

#####L_1 NORMALIZATION
##def l1_normalization(X):
##    q = np.abs(X).sum(1, keepdims = True)
##    q = np.maximum(q, 1e-15)
##    X = X / q
##    return X
##
##def whitening(Xtrain , Xval):
##    mu = Xtrain.mean(0, keepdims = True)
##    sigma = np.cov(Xtrain.T)
##    evals , evecs = np.linalg.eigh(sigma)
##    w = evecs/np.sqrt(evals)
##    Xtrain = (Xtrain - mu) @ w
##    Xval = (Xval - mu) @ w
##    return Xtrain , Xval
##   
    
   

Xtrain, Ytrain = load_reshape("train.npz")
print("Training set after reshape: ", Xtrain.shape, Ytrain.shape)
Xtest, Ytest = load_reshape("test.npz")
print("Test set after reshape: ", Xtest.shape, Ytest.shape)

##Xtrain = l2_normalization(Xtrain)
##Xtest = l2_normalization(Xtest)
##Xtrain, Xtest = whitening(Xtrain, Xtest)

""" PCA """

#Number of principal components
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

""" PCA using sklearn"""
"""
from sklearn.decomposition import PCA

# Initialize and fit PCA
pca = PCA(n_components=k)
Xtrain = pca.fit_transform(Xtrain)
Xtest = pca.fit_transform(Xtest)
print("Shape of Training and Test set after PCA:", Xtrain.shape, Xtest.shape)
"""



""" Recursive feature elimination """
### Requires Validation set

##def recursive_feature_elimination(Xtrain, Ytrain, Xval, Yval):
##    n = Xtrain.shape[1]
##    # Start by using all the features
##    best_features = np.ones(n, dtype=np.bool)
##    params = train(Xtrain, Ytrain)
##    labels = inference(Xval, params)
##    best_accuracy = (labels == Yval).mean()
##    while True:
##        improved = False
##        features = best_features.copy()
##        for j in features.nonzero()[0]:
##            # Evaluate the removal of feature j
##            features[j] = False
##            params = train(Xtrain[:, features], Ytrain)
##            labels = inference(Xval[:, features], params)
##            accuracy = (labels == Yval).mean()
##            if accuracy > best_accuracy:
##                best_accuracy = accuracy
##                best_features = features.copy()
##                improved = True
##                features[j] = True
##                # Stop when no improvement is obtained
##            if not improved:
##                return best_features


##print(Xtrain.shape[1])



###0 hidden
##net = pvml.MLP([Xtrain.shape[1], 10])

### 1 hidden: 88
##nhidden = int(np.sqrt(10*Xtrain.shape[1]))
##print("Number of hidden neurons:", nhidden)
##net = pvml.MLP([Xtrain.shape[1], nhidden, 10])

# 2 hidden: 181, 42
nhidden2 = int(np.cbrt(Xtrain.shape[1]*100))
nhidden1 = int(np.sqrt(Xtrain.shape[1]*nhidden2))
print("Number of hidden neurons:", nhidden1, nhidden2)
net = pvml.MLP([Xtrain.shape[1], nhidden1, nhidden2, 10])

### 3 hidden: 263 , 88, 29
##nhidden31 = int(np.sqrt(Xtrain.shape[1]*np.sqrt(10*Xtrain.shape[1])))
##nhidden32 = int(np.sqrt(10*Xtrain.shape[1]))
##nhidden33 = int(np.power(1000*Xtrain.shape[1], (1/4)))
##print("Number of hidden neurons:", nhidden31, nhidden32, nhidden33)
##net = pvml.MLP([Xtrain.shape[1], nhidden31, nhidden32, nhidden33, 10])


### TRAINING
m = Ytrain.size

### activate interacting mode with ion()
plt.ion()
train_accs = []
test_accs = []
epochs = []
batch_size = 10

print("Training: starting")
### what if the number of epochs changes
for epoch in range(51):
    # parameters: training data and learning rate
    # using SGD 
    net.train(Xtrain, Ytrain, 1e-4, batch = batch_size, steps = m // batch_size)
    if epoch % 5 == 0:
      # return predictions and probability
      train_acc = accuracy(net, Xtrain, Ytrain)
      test_acc = accuracy(net, Xtest, Ytest)
      print("Epoch number", epoch, "  Training Accuracy:", train_acc, "Test accuracy:", test_acc)
    
      train_accs.append(train_acc)
      test_accs.append(test_acc)
      epochs.append(epoch)
    
      plt.clf() # clear the plots
      plt.plot(epochs, train_accs)
      plt.plot(epochs, test_accs)
      plt.title(f'Network Neurons per layer: ({Xtrain.shape[1]}, {nhidden1}, {nhidden2}, 10)' )
      plt.xlabel("Epoch")
      plt.ylabel("Accuracy [%]")
      plt.legend(['train', 'test'])
      plt.pause(0.01) # stops for a given amount of time (even smaller)
      # common to save after each epoch
      plt.savefig('mlp(2hiddentest).png')
      net.save("mlp(2hiddentest).npz")

print("Training: finished")

##print("Epoch number", epochs[49], "Training Accuracy:", train_accs[49], "Test accuracy:", test_accs[49])

plt.ioff() # interactive off
plt.show() # keep the window open
