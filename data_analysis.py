import numpy as np
import matplotlib.pyplot as plt

n = int(input("Which data you want to visualize? \n"))
n = int(n)
##n = 23

words = ["Tshirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","AnkleBoot"]

Xtrain , Ytrain = np.load("train.npz").values()
Xtest , Ytest = np.load("test.npz").values()


image = Xtest[n, :]
plt.imshow(image,cmap='gray')
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
plt.title(words[Ytest[n]] + f"  (Data element: {n} , value = ({Ytest[n]}))" )
plt.colorbar()
plt.show()












