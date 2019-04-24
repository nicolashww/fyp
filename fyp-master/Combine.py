# Importing Modules
import numpy as np
import pickle 

savePath = 'D:/fyp-master/TrainingData/'
productName = 'DF94'
productName2 = 'AC8C'
bandName = 'Intensity_VH_S'

pixelData = pickle.load(open((savePath + productName + '_' + bandName + '_collocateddata'), 'rb'))
pixelData2 = pickle.load(open((savePath + productName2 + '_' + bandName + '_collocateddata'), 'rb'))
labels = pickle.load(open((savePath + productName + '_' + bandName + '_collocatedlabels'), 'rb'))
labels2 = pickle.load(open((savePath + productName2 + '_' + bandName + '_collocatedlabels'), 'rb'))

size = len(pixelData)
size2 = len(pixelData2)

pixelData = pixelData + pixelData2
labels = labels + labels2

if(size + size2 == len(pixelData)):
    print("Saved")
    
pickle.dump(pixelData, open(savePath + productName + '_' + productName2 + '_' + bandName + '_collocateddata', 'wb'))
pickle.dump(labels, open(savePath + productName + '_' + productName2 + '_' + bandName + '_collocatedlabels', 'wb'))




