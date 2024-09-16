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

###L_1 NORMALIZATION
def l1_normalization(X):
    q = np.abs(X).sum(1, keepdims = True)
    q = np.maximum(q, 1e-15)
    X = X / q
    return X

def whitening(Xtrain , Xval):
    mu = Xtrain.mean(0, keepdims = True)
    sigma = np.cov(Xtrain.T)
    evals , evecs = np.linalg.eigh(sigma)
    w = evecs/np.sqrt(evals)
    Xtrain = (Xtrain - mu) @ w
    Xval = (Xval - mu) @ w
    return Xtrain , Xval
   
    
   

Xtrain, Ytrain = load_reshape("train.npz")
print("Training set after reshape: ", Xtrain.shape, Ytrain.shape)
Xtest, Ytest = load_reshape("test.npz")
print("Test set after reshape: ", Xtest.shape, Ytest.shape)

##Xtrain = l1_normalization(Xtrain)
##Xtest = l1_normalization(Xtest)
##Xtrain, Xtest = whitening(Xtrain, Xtest)

""" Divide Training set """
from sklearn.model_selection import train_test_split

testsize = 0.4 #percentuale messa in validation
Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=testsize, random_state=42)

print("Training set shapes" , Xtrain.shape, Ytrain.shape)
print("Validation set shapes" ,Xval.shape, Yval.shape)

""" PCA """
### Number of principal components
##k = 459
##print("Number of principal components:", k)
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





""" Recursive feature elimination """
### Requires Validation set

def recursive_feature_elimination(Xtrain, Ytrain, Xval, Yval):
    n = Xtrain.shape[1]
    # Start by using all the features
    best_features = np.ones(n, dtype=np.bool)
    params = train(Xtrain, Ytrain)
    labels = inference(Xval, params)
    best_accuracy = (labels == Yval).mean()
    while True:
        improved = False
        features = best_features.copy()
        for j in features.nonzero()[0]:
            # Evaluate the removal of feature j
            features[j] = False
            params = train(Xtrain[:, features], Ytrain)
            labels = inference(Xval[:, features], params)
            accuracy = (labels == Yval).mean()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_features = features.copy()
                improved = True
                features[j] = True
                # Stop when no improvement is obtained
            if not improved:
                return best_features




def knn_inference(X, Xtrain, Ytrain, k):
    classes = Ytrain.max() + 1
    D = _dist_matrix(X, Xtrain)
    neighs = np.argpartition(D, k, 1)[:, :k]
    counts = _bincount_rows(Ytrain[neighs], classes)
    labels = np.argmax(counts, 1)
    return labels

def _bincount_rows(X, values):
    """Compute one histogram for each row."""
    # np.bincount works only on 1D arrays.
    # This extends it to 2D arrays.
    m = X.shape[0]
    idx = X.astype(int) + values * np.arange(m)[:, np.newaxis]
    c = np.bincount(idx.ravel(), minlength=values * m)
    return c.reshape(-1, values)

def _dist_matrix(X1, X2):
    """Compute the matrix of all squared distances."""
    Q1 = (X1 ** 2).sum(1, keepdims=True)
    Q2 = (X2 ** 2).sum(1, keepdims=True)
    return Q1- 2 * X1 @ X2.T + Q2.T


def knn_select_k(X, Y, maxk=101):
    """Leave-one-out selection of the number of neighbors."""
    D = _dist_matrix(X, X)
    classes = Y.max() + 1
    np.fill_diagonal(D, np.inf)
    neighs = np.argsort(D, 1)
    best_k = 1
    best_acc =-1
    for k in range(1, maxk + 1):
        counts = _bincount_rows(Y[neighs[:, :k]], classes)
        labels = np.argmax(counts, 1)
        accuracy = (labels == Y).mean()
        if accuracy > best_acc:
            best_acc = accuracy
            best_k = k
    return best_k

bestk = 4
##bestk = knn_select_k(Xtrain, Ytrain)
print("Best k value: ", bestk)

predictions = knn_inference(Xtrain, Xtrain, Ytrain, bestk)
Tpredictions = knn_inference(Xtest, Xtrain, Ytrain, bestk)

accuracy = (predictions == Ytrain).mean()    
Taccuracy = (Tpredictions == Ytest).mean()

print("Training accuracy:", accuracy * 100)
print("Test accuracy:", Taccuracy * 100)


