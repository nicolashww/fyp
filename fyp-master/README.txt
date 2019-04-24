Collocating radar data to get model
Step 1: Get (x,y) of boundary of radar image on biomass map
Step 2: Create polygon mask
Step 3: Crop out the region of interest from biomass map
Step 4: Get (x,y) of masked pixels on biomass map
Step 5: Get pixel information using (x,y) and append with label
Step 6: Train model 

GetMask.py --> Step 1

GetCollocatedTrainingData.py --> Step 2, 3, 4, 5

SupervisedCollocated.py --> Step 6

Raw radar data to get model
Step 1: Get (x,y) of boundary of radar image on biomass map
Step 2: Create polygon mask
Step 3: Crop out the region of interest from biomass map
Step 4: Get (x,y) of masked pixels on biomass map
Step 5: Get latlon of these (x,y) masked pixels
Step 6: Map them to radar image using latlon and get (x,y) of these pixels on radar image
Step 7: Get pixel information from mapped (x,y) and append with label
Step 8: Train model

GetMask.py --> Step 1

GetRawTrainingData.py --> Step 2, 3, 4, 5, 6, 7

SupervisedRaw.py --> Step 8

GeoPos[lat, lon], lat is smaller number
PixPos[x, y]

What data do you use
Variance
KNN confidence?
kfold
