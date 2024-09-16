import numpy as np
import matplotlib.pyplot as plt

Xtrain, Ytrain = np.load("train.npz").values()
#Xtest, Ytest = np.load("test.npz").values()

print(Xtrain.shape, Ytrain.shape)
#print(Xtest.shape, Ytest.shape)

Xtrain = Xtrain.reshape(Xtrain.shape[0], -1)
#we want to keep the first dimension (one row for each data element)
#we reshape the other 2 dimensions (28x28)
#Xtest = Xtest.reshape(Xtest.shape[0], -1) 
#print(Xtrain.shape, Xtest.shape)

def one_hot(vector, n_classes):
  return np.squeeze(np.eye(n_classes)[vector.reshape(-1)])

def multinomial_logreg_inference(X, W, b):
    logits = X @ W + b.T 
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return probs

def multinomial_logreg_train(X, Y, lr=1e-3, steps=1000):
    Loss = []
    Step = [] 
    m, n = X.shape 
    k = Y.max() + 1 # number of classes
    data = np.load("Mult10000.npz")
    W = data["arr_0"]
    b = data["arr_1"]
##    W = np.zeros((n, k))
##    b = np.zeros(k)
    H = np.zeros((m, k))
    H[np.arange(m), Y] = 1
    Y = one_hot(Y, 10)
    for step in range(steps):
        P = multinomial_logreg_inference(X, W, b)
        P = np.clip(P,0.0001,0.9999)
        #clip limits the values
        grad_W = (X.T @ (P- H)) / m
        grad_b = (P- H).mean(0)
        W-= lr * grad_W
        b-= lr * grad_b
        loss = (- Y*np.log(P)).mean()
        Loss.append(loss)
        newstep = step + 5000 
        Step.append(newstep)
        if step % (steps/(20)) == 0:
            print(step/steps*100, "%")
            plt.clf()
            plt.title("Loss function")
            plt.plot(Step, Loss)
            #plt.plot(eps, test_accs)
            plt.legend(["train", "test"])
            plt.xlabel("Steps")
            plt.ylabel("Loss function(%)")
            plt.pause(0.01)
            plt.savefig('Lossfunction(Multlogreg10000).png')
            np.savez_compressed("Mult10000.npz", W, b)

        
    return W,b

##Steppers = (50,100,500,1000,2000,5000,10000)
steps = 5000
##for steps in Steppers:
Steps = steps
print(f"Starting Multinomial Logistic regression with {Steps} steps")
W,b = multinomial_logreg_train(Xtrain, Ytrain, 1e-4, Steps)
np.savez_compressed(f"Mult{Steps}.npz", W, b)
print("End of training")

plt.ioff()
plt.show()

##
##if epoch % 1000 == 0:
##        predictions, probs = mlp.inference(X)
##        train_acc = 100 * (predictions == Y).mean()
##        predictions, probs = mlp.inference(Xtest)
##        test_acc = 100 * (predictions == Ytest).mean()
##        print(f"{epoch} train:{train_acc:3.1f}% test : {test_acc:3.1f}%")
##        #3.1f 3 cifre, di cui una float (the format)
##        eps.append(epoch)
##        train_accs.append(train_acc)
##        test_accs.append(test_acc)
##        plt.clf()
##        plt.title(f"Accuracy with {name} features")
##        plt.plot(eps, train_accs)
##        plt.plot(eps, test_accs)
##        plt.legend(["train", "test"])
##        plt.xlabel("epoch")
##        plt.ylabel("accuracy (%)")
##        plt.pause(0.01)










