# Importing Modules
import matplotlib.pyplot as plt
import numpy as np
import pickle 
from snappy import ProductIO
#from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

def print_confusion_matrix(y_test, y_pred):
    from sklearn.metrics import confusion_matrix
    import numpy
    outstr = ""
    con_matrix = confusion_matrix(y_test, y_pred)
    outstr += "Confusion Matrix (non-normalized):\n" + str(con_matrix)
    #outstr += "\nConfusion Matrix (normalized):\n"
    #outstr += str(con_matrix.astype('float') / con_matrix.sum(axis=1)[:, numpy.newaxis])
    #outstr += "\nCalculated accuracy: " + str(numpy.sum(numpy.diagonal(con_matrix))/ numpy.sum(con_matrix))
    print(outstr)
    return outstr

modelSavePath = 'D:/fyp-master/Models/'
savePath = 'D:/fyp-master/TrainingData/'
productName = 'DF94'
productName2 = 'AC8C'
bandName = 'Intensity_VH_S'
bandType = 'VH'

# Prepare radar image
r = ProductIO.readProduct('D:/Test/AGB_2015_0.zip')
bioBand = r.getBand('band_1')
bioW = bioBand.getRasterWidth()
bioH = bioBand.getRasterHeight()

# Load data and labels
pixelData = pickle.load(open((savePath + productName + '_' + bandName + '_collocateddata'), 'rb'))
labels = pickle.load(open((savePath + productName + '_' + bandName + '_collocatedlabels'), 'rb'))
#pixelData1 = pickle.load(open((savePath + productName2 + '_' + bandName + '_newcollocateddata'), 'rb'))
#labels1 = pickle.load(open((savePath + productName2 + '_' + bandName + '_newcollocatedlabels'), 'rb'))
#pixelData = pickle.load(open((savePath + productName + '_' + productName2 + '_' + bandName + '_collocateddata'), 'rb'))
#labels = pickle.load(open((savePath + productName + '_' + productName2 + '_' + bandName + '_collocatedlabels'), 'rb'))
#pixelData = pixelData + pixelData1
#labels = labels + labels1
data_train, data_test, label_train, label_test = train_test_split(pixelData, labels, test_size=0.2)

data_train = np.reshape(data_train, (-1, 1))
data_test = np.reshape(data_test, (-1, 1))

print("For product " + productName + " " + bandType)

#pixelData = np.reshape(pixelData, (-1, 1))
#labels = np.reshape(labels, (-1, 1))
index = []
accu = []
# Create Random Forest Model

for i in range (2, 20):
    model = RandomForestClassifier(n_estimators=i)
    model.fit(data_train, label_train)
    predicted = model.predict(data_test)
    print_confusion_matrix(predicted, label_test)
    accuracy = accuracy_score(label_test, predicted)
    print("Random Forest accuracy for n=",i, "is", accuracy)

# Load Random Forest Model
#model = pickle.load(open(('C:/Users/ASD/Desktop/FYP/fyp-master/Models/DF94_AC8C_Intensity_VH_S_RF'), 'rb'))
#predicted = model.predict(pixelData)
#accuracy = accuracy_score(labels, predicted)

# Save Random Forest Model
#pickle.dump(model, open(modelSavePath + productName + '_' + bandName + '_RF', 'wb'))
#pickle.dump(model, open(modelSavePath + productName + '_' + productName2 + '_' + bandName + '_RF', 'wb'))

# Save KNN
#modelKnn = pickle.load(open(('C:/Users/ASD/Desktop/FYP/fyp-master/Models/DF94_AC8C_Intensity_VH_S_KNN'), 'rb'))
knnindex = []
kaccu = []
for i in range(1,40): 
    knn = KNeighborsClassifier(n_neighbors=i)
    modelKnn = knn.fit(data_train, label_train)
    predKnn = modelKnn.predict(data_test)
    print_confusion_matrix(predKnn, label_test)
    knnAccu = accuracy_score(label_test, predKnn)
    print("KNN accuracy for k =", i, "is", knnAccu)
    #knnindex.append(i)
    #kaccu.append(knnAccu)

#c = plt.scatter(knnindex,kaccu)
#plt.axis([0, 60, 0.5, 0.65])
#plt.show()
# Save KNN
#pickle.dump(modelKnn, open(modelSavePath + productName + '_' + bandName + '_KNN', 'wb'))
#pickle.dump(modelKnn, open(modelSavePath + productName + '_' + productName2 + '_' + bandName + '_KNN', 'wb'))
