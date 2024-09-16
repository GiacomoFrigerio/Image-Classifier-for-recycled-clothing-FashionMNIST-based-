import numpy as np
import matplotlib.pyplot as plt
import pvml
import os

#we also return scores to verify which are the result with bigger confidence found by the system
##def inference(X, w, b):
##    scores = X @ w + b
##    labels = (scores > 0).astype(int)
##    return labels, scores
def load_reshape(path):
    X, Y = np.load(path).values()
    X = X.reshape(X.shape[0], -1)
    return X, Y

##X, Y = load_reshape("train.npz")
##print("Training set after reshape: ", X.shape, Y.shape)
Xtestt, Ytestt = load_reshape("test.npz")
##print("Test set after reshape: ", Xtestt.shape, Ytestt.shape)

X, Y = np.load("train.npz").values()
##Xtest, Ytest = np.load("test.npz").values()
Xtest, Ytest = load_reshape("test.npz")


neuralnet = "mlp(2hidden).npz"
net = pvml.MLP.load(neuralnet)

##w = net.weights[0]
##b = net.biases[0]
##predictions,scores = net.inference(Xtest)
##min = [99,99,99]
##max = [0,0,0]
##pos_ind = []
##neg_ind = []


numberofclasses = 10
words = ["Tshirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","AnkleBoot"]

predictions, probs = net.inference(Xtest)
acc = 100 * (predictions == Ytest).mean()
print("Test accuracy: ", acc)

classes = ["Tshirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","AnkleBoot"]
classes.sort()

max = np.array([[0,0,0],[0,0,0],[0,0,0]]).astype(float)
for j in range(predictions.size):
    if predictions[j] != Ytest[j]:
            if all(x < probs[j,predictions[j]] for x in max[:,0]):
                print(probs[j, predictions[j]])
                max[2,0] = probs[j,predictions[j]]
                max[2,1] = j
                max[2,2] = predictions[j]
                max = max[np.argsort(max[:,0]),:][::-1]

print(max)
            
    # specifichiamo le predizioni sbagliate
    
##for m in max:
##     files = os.listdir("images/test/" +classes[int(m[1]//20)] )
##     print("images/test/" + classes[int(m[1]//20)]+"/"+files[int(m[1]%20)])
##     print("Ind folder: ", classes[int(m[1]//20)], " Ind file: ", files[int(m[1]%20)], " Wrongly predicted class: ", classes[int(m[2])])
##    

















##for j in range(scores.size):
##    if predictions[j] != Y[j]:
##        if predictions[j] == 0:
##             if all(x > scores[j] for x in min):
##                min[2] = scores[j]
##                min.sort()
##                pos_ind.insert(0, j)
##        if predictions[j] == 1:
##            if all(x < scores[j] for x in max):
##                max[2] = scores[j]
##                max.sort(reverse=True)
##                neg_ind.insert(0, j - 6250)
##
##
##
##
##number_rev = 3 #number of reviews printed
##
##print("Worse predictions on the test set for negative values:")
##for file_index in pos_ind[:number_rev]:
##    print("Score: ", scores[file_index], "\n")
##    files = os.listdir("aclImdb/test/pos")
##    f = open("aclImdb/test/pos/" + files[file_index])
##    print(f.read())
##    f.close()
##    print("\n")
##
##print("Worse prediction on the test set for positive values")
##for file_index in neg_ind[:number_rev]:
##    print("Score: ", scores[file_index + 6250], "\n")
##    files = os.listdir("aclImdb/test/neg")
##    f = open("aclImdb/test/neg/" + files[file_index])
##    print(f.read())
##    f.close()
##    print("\n")
