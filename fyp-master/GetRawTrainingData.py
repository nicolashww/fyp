import pickle
import numpy as np
from snappy import jpy
from snappy import ProductIO
from snappy import WKTReader
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt 

# Returns lat/lon from (x, y)
def getGeoFromPixel(geo_coding, pixPos):
        latLon = []
        for x in range (0, len(pixPos)):
                geoPos = geo_coding.getGeoPos(PixelPos(float(pixPos[x][0]), float(pixPos[x][1])), None)
                latLon.append([geoPos.lat, geoPos.lon])
        return latLon

# Returns (x, y) from lat/lon
def getPixelFromGeo(geo_coding, geoPos):
        pix = []
        for x in range (0, len(geoPos)):
                pixPos = geo_coding.getPixelPos(GeoPos(float(geoPos[x][0]), float(geoPos[x][1])), None)
                xCoord = np.round(pixPos.getX())
                yCoord = np.round(pixPos.getY())
                if (xCoord >= 0) and (yCoord >= 0):
                        pix.append([xCoord, yCoord])
        return pix

# Preparation
productName = '5007'
maskPath = 'D:/fyp-master/Polygon/'
#fileName = 'D:/Test/collocate_' + productName + '.dim'
fileName = 'D:/Test/' + productName + '.dim'
bandName = 'Intensity_VV'
savePath = 'D:/fyp-master/TrainingData/'

# jpy conversion
PixelPos = jpy.get_type('org.esa.snap.core.datamodel.PixelPos')
GeoPos = jpy.get_type('org.esa.snap.core.datamodel.GeoPos')
GeoCoding = jpy.get_type('org.esa.snap.core.datamodel.GeoCoding')
RasterDataNode = jpy.get_type('org.esa.snap.core.datamodel.RasterDataNode')

# Prepare French Guyana .tif
p = ProductIO.readProduct('D:/Test/AGB_2015_0.zip')
bioBand = p.getBand('band_1')
bioW = bioBand.getRasterWidth()
bioH = bioBand.getRasterHeight() 
bio_data = np.zeros(bioW * bioH, np.float32)  #Return a new array of given shape and type, filled with zeros. Filled only x-ways. np.zeros(5) = {0,0,0,0,0}
bioBand.readPixels(0, 0, bioW, bioH, bio_data) #readPixels(x,y,w,h, Array) x : x offset of upper left corner. y : y offset of upper left corner. w : width. h : height. Array : output array
bio_data.shape = bioH, bioW

# Prepare radar image
r = ProductIO.readProduct(fileName)
radarBand = r.getBand(bandName)
radarW = radarBand.getRasterWidth()
radarH = radarBand.getRasterHeight()

# Initilization of raster band and geocoding for French Guyana and radar image
raster_Band = jpy.cast(bioBand, RasterDataNode)
raster_BandRadar = jpy.cast(radarBand, RasterDataNode)
geo_CodingBio = jpy.cast(raster_Band.getGeoCoding(), GeoCoding)    #get geocoding of biomass map
geo_CodingRadar = jpy.cast(raster_BandRadar.getGeoCoding(), GeoCoding) #get geocoding of radar image

# Create mask of ROI on French Guyana .tif
polygon = pickle.load(open((maskPath + productName + 'Mask'), 'rb'))

img = Image.new('L', (bioW, bioH), 0)
ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
mask = np.array(img)
imgplot = plt.imshow(mask)
imgplot.write_png(productName + 'Mask.png')

# Retrieve (x,y) coordinates of pixels within mask of French Guyana .tif
coordsH = []
coordsM = []
coordsL = []
result = np.zeros(bioW * bioH, np.float32)
result.shape = bioH, bioW

for y in range (0, bio_data.shape[0]):  #shape[0] = 401 (height)
       for x in range (0, bio_data.shape[1]):
                if mask[y][x] == 1 and bio_data[y][x] >= 0:
                        result[y][x] = bio_data[y][x]
                        if bio_data[y][x] >= float(350):
                                coordsH.append([x, y])
                        elif bio_data[y][x] >= float(290):
                                coordsM.append([x, y]) 
                        elif bio_data[y][x] >= float(0):
                                coordsL.append([x, y])         
                else:
                        result[y][x] = -9999

imgplot = plt.imshow(result)
imgplot.write_png(productName + 'Result.png')

# Get lat and lon of relevant pixels of French Guyana .tif to map to radar image
lonLatH = []
lonLatM = []
lonLatL = []

lonLatH = getGeoFromPixel(geo_CodingBio, coordsH)
lonLatM = getGeoFromPixel(geo_CodingBio, coordsM)
lonLatL = getGeoFromPixel(geo_CodingBio, coordsL)

# Find (x,y) coordinates of pixels in radar image from lat and lon
radarCoordsH = []
radarCoordsM = []
radarCoordsL = []

radarCoordsH = getPixelFromGeo(geo_CodingRadar, lonLatH)
radarCoordsM = getPixelFromGeo(geo_CodingRadar, lonLatM)
radarCoordsL = getPixelFromGeo(geo_CodingRadar, lonLatL)

# Get pixel data from radar image using (x,y) coordinates
labels = []
pixelData = []
originalOOR = 0
countOOR = 0
total = 0
naN = 0

raster_BandRadar.loadRasterData()

# Get pixel data for high biomass concentration
for x in range (0, len(radarCoordsH)):
        if(radarCoordsH[x][0] >= radarW) or (radarCoordsH[x][1] >= radarH):
                originalOOR += 10000
        else:
                for y in range (0, 100):
                        for x in range (0, 100):
                                if (int(radarCoordsH[x][0] + x) >= radarW) or (int(radarCoordsH[x][1] + y) >= radarH):
                                        countOOR += 1
                                else:
                                        pix = raster_BandRadar.getPixelFloat(int(radarCoordsH[x][0] + x), int(radarCoordsH[x][1] + y))   
                                        if (pix > 0):
                                                pixelData.append(pix)
                                                labels.append(2)
                                        else:
                                                naN += 1

# Checks if pixel data has been successfully appended
if (originalOOR + countOOR + naN + len(pixelData) == len(radarCoordsH) * 10000):
        print("Pixel results tally for H")

# Get pixel data for medium biomass concentration
for x in range (0, len(radarCoordsM)):
        if(radarCoordsM[x][0] >= radarW) or (radarCoordsM[x][1] >= radarH):
                originalOOR += 10000
        else:
                for y in range (0, 100):
                        for x in range (0, 100):
                                if (int(radarCoordsM[x][0] + x) >= radarW) or (int(radarCoordsM[x][1] + y) >= radarH):
                                        countOOR += 1
                                else:
                                        pix = raster_BandRadar.getPixelFloat(int(radarCoordsM[x][0] + x), int(radarCoordsM[x][1] + y))   
                                        if (pix > 0):
                                                pixelData.append(pix)
                                                labels.append(1)
                                        else:
                                                naN += 1

# Checks if pixel data has been successfully appended so far
if (originalOOR + countOOR + naN + len(pixelData) == len(radarCoordsH) * 10000 + len(radarCoordsM) * 10000):
        print("Pixel results tally for M")

# Get pixel data for low biomass concentration
for x in range (0, len(radarCoordsL)):
        if(radarCoordsL[x][0] >= radarW) or (radarCoordsL[x][1] >= radarH):
                originalOOR += 10000
        else:
                for y in range (0, 100):
                        for x in range (0, 100):
                                if (int(radarCoordsL[x][0] + x) >= radarW) or (int(radarCoordsL[x][1] + y) >= radarH):
                                        countOOR += 1
                                else:
                                        pix = raster_BandRadar.getPixelFloat(int(radarCoordsL[x][0] + x), int(radarCoordsL[x][1] + y))   
                                        if (pix > 0):
                                                pixelData.append(pix)
                                                labels.append(0)
                                        else:
                                                naN += 1

# Checks if pixel data has been successfully appended altogether
if (originalOOR + countOOR + naN + len(pixelData) == len(radarCoordsH) * 10000 + len(radarCoordsM) * 10000 + len(radarCoordsL) * 10000):
        print("Pixel results tally for L")

# Save data and labels
pickle.dump(pixelData, open(savePath + productName + '_' + bandName + '_rawdata', 'wb'))
pickle.dump(labels, open(savePath + productName + '_' + bandName + '_rawlabels', 'wb'))
print("Data saved")
# Cleans up products
p.dispose()
r.dispose()