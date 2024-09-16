import numpy as np
import matplotlib.pyplot as plt
import image_features

Xtrain, Ytrain = np.load("train.npz").values()
Xtest, Ytest = np.load("test.npz").values()

print(Xtrain.shape, Ytrain.shape)
print(Xtest.shape, Ytest.shape)

##Xtrain = Xtrain.reshape(Xtrain.shape[0], -1)

#we want to keep the first dimension (one row for each data element)
#we reshape the other 2 dimensions (28x28)
#Xtest = Xtest.reshape(Xtest.shape[0], -1) 
#print(Xtrain.shape, Xtest.shape)

''' INPUT FEATURE VALUES '''

featurename = []
name = ""
color = input("Input yes if you want Color Histogram feature or No if not: \n")
if color == "yes" :
    featurename += ["color"]
    name += "color"
cooc = input("Input yes if you want Cooccurrence Matrix feature or No if not: \n")
if cooc == "yes" :
    featurename += ["cooc"]
    name += "cooc"
rgb = input("Input yes if you want RGB Cooccurrence Matrix feature or No if not: \n")
if rgb == "yes" :
    featurename += ["rgb"]
    name += "rgb"
edge = input("Input yes if you want Edge Direction feature or No if not: \n")
if edge == "yes" :
    featurename += ["edge"]
    name += "edge"

##print(featurename)
##print(name)

def process_directory(Xchange):
    all_features = []
##    all_labels = []
##    klass_label = 0
##    for klass in classes:
##        image_files = os.listdir(path + "/" + klass)
    for i in range(Xchange.shape[0]):
##            image_path = path + "/" + klass + "/" + imagename
##            image = plt.imread(image_path) / 255.0
        #we get a uint8 (8 bit per pixel) which is not very good: it is better
        #to use floats, so we divide by 255.0 (also ranging pixels between 0 and 1).
##            print(image.shape, image.dtype)
##            plt.imshow(image)
##            plt.show() #we print the image to be sure everything works well
##            return
            

            if "cooc" in featurename:
                features1 = image_features.cooccurrence_matrix(Xtrain[i, :])
                features1 = features1.reshape(-1)
            if "color" in featurename:
                features2 = image_features.color_histogram(Xtrain[i,:])
                features2 = features2.reshape(-1)
            if "edge" in featurename:   
                features3 = image_features.edge_direction_histogram(image)
                features3 = features3.reshape(-1)
            if "rgb" in featurename:
                features4 = image_features.rgb_cooccurrence_matrix(image)
                features4 = features4.reshape(-1)


            if "cooc" in featurename and "color" in featurename and "edge" in featurename and "rgb" in featurename :
                features = np.concatenate([features1,features2,features3,features4])
                
            if "cooc" not in featurename and "color" in featurename and "edge" in featurename and "rgb" in featurename :
                features = np.concatenate([features2,features3,features4])
            if "cooc" in featurename and "color" not in featurename and "edge" in featurename and "rgb" in featurename :
                features = np.concatenate([features1,features3,features4])
            if "cooc" in featurename and "color" in featurename and "edge" not in featurename and "rgb" in featurename :
                features = np.concatenate([features1,features2,features4])
            if "cooc" in featurename and "color" in featurename and "edge" in featurename and "rgb" not in featurename :
                features = np.concatenate([features1,features2,features3])
                
            if "cooc" not in featurename and "color" not in featurename and "edge" in featurename and "rgb" in featurename :
                features = np.concatenate([features3,features4])
            if "cooc" not in featurename and "color" in featurename and "edge" not in featurename and "rgb" in featurename :
                features = np.concatenate([features2,features4])
            if "cooc" not in featurename and "color" in featurename and "edge" in featurename and "rgb" not in featurename :
                features = np.concatenate([features2,features3])
            if "cooc" in featurename and "color" not in featurename and "edge" not in featurename and "rgb" in featurename :
                features = np.concatenate([features1,features4])
            if "cooc" in featurename and "color" not in featurename and "edge" in featurename and "rgb" not in featurename :
                features = np.concatenate([features1,features3])
            if "cooc" in featurename and "color" in featurename and "edge" not in featurename and "rgb" not in featurename :
                features = np.concatenate([features1,features2])
                

            if "cooc" in featurename and "color" not in featurename and "edge" not in featurename and "rgb" not in featurename :
                features = features1
            if "cooc" not in featurename and "color" in featurename and "edge" not in featurename and "rgb" not in featurename :
                features = features2
            if "cooc" not in featurename and "color" not in featurename and "edge" in featurename and "rgb" not in featurename :
                features = features3
            if "cooc" not in featurename and "color" not in featurename and "edge" not in featurename and "rgb" in featurename :
                features = features4
                
 

            

            


            #this procedure extracts features from images, for example the histogram of different colors
            

#we try different feature extraction (we can find them in image_features.py)
#color_histogram
#cooccurrence_matrix
#rgb_cooccurrence_matrix            
#edge_direction_histogram
            #we reshape it because color_histogram gives us 3 different vectors with 64 elements
            #but we want them to be in a single vector 
##            plt.bar(range(192), features) #histogram to see features
##            plt.show()
##            print(features.shape)

            #we now append everything to have a feature matrix
            all_features.append(features)
##            all_labels.append(klass_label)
##        klass_label += 1 #to give a different numeric label to each class
    X = np.stack(all_features, 0)
##    Y = np.array(all_labels)
    return X




X = process_directory(Xtrain)
Y = Ytrain
print("train", X.shape,Y.shape)
np.savez_compressed(f"trainmult({name}).npz", X,Y)

X = process_directory(Xtest)
Y = Ytest
print("test", X.shape,Y.shape)
np.savez_compressed(f"testmult({name}).npz", X,Y)


'''
np.savez_compressed("trainCOC.npz", X,Y)

X,Y = process_directory("images/test")
print("test", X.shape, Y.shape)
np.savez_compressed("testCOC.npz", X,Y)
'''
