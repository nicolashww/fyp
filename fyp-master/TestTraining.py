# Importing Modules
#import matplotlib.pyplot as plt
import numpy as np
import pickle 
#from sklearn.cluster import KMeans
#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold

def get_k_fold_accuracy(model, folds, data, labels, print_out=False):
    data_train = []
    data_test = []

    print("Calculating KFold accuracy with {} iterations".format(folds))

    kf = KFold(len(data), n_folds=folds, shuffle=True)
    print(len(data))
    total_accuracy = 0

    for training_indices, testing_indices in kf:
            
        data_train = np.reshape(data_train, (-1, 1))
        data_test = np.reshape(data_test, (-1, 1))
        model.fit(data_train, label_train)
        
        y_predicted = model.predict(data_test)
        total_accuracy += accuracy_score(label_test, y_predicted)
    return total_accuracy/folds

savePath = 'C:/Users/ASD/Desktop/FYP/fyp-master/TrainingData/'
productName = 'DF94'
bandName = 'Intensity_VH_S'

pixelData = pickle.load(open((savePath + productName + '_' + bandName + '_collocateddata'), 'rb'))
labels = pickle.load(open((savePath + productName + '_' + bandName + '_collocatedlabels'), 'rb'))

# Random Forest
model = RandomForestClassifier()
accuracy = get_k_fold_accuracy(model, 5, pixelData, labels, print_out=True)

print("Random Forest accuracy: ", accuracy)


'''
# KNN
for i in range(1,30): 
    knn = KNeighborsClassifier(n_neighbors=i)
    modelKnn = knn.fit(data_train, label_train)
    predKnn = modelKnn.predict(data_test)
    knnAccu = accuracy_score(label_test, predKnn)
    print("KNN accuracy for ", i, " is,", knnAccu)

colors = ["g.", "r.", "c."]

def save_model(name, model):
    filename = name +'.sav'
    pickle.dump(model, open(outdir + '/' + filename, 'wb'))

def load_model(path):
    model = pickle.load(open(path, 'rb'))

for i in range(len(x)):
    print("coordinates:", x[i], "label:", labels[i])
    plt.plot(x[i][0], colors[labels[i]], markersize = 10)

plt.scatter(centroids[:, 0], centroids[:,], marker = "x", s = 150, linewidths = 5, zorder = 10)

plt.show()
'''