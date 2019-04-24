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
                pix.append([xCoord, yCoord])
        return pix

# Preparation
productName = 'AC8C'
maskPath = 'D:/fyp-master/Polygon/'
fileName = 'D:/Test/collocate_' + productName + '.dim'
bandName = 'Intensity_VH_S'
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
geo_Coding = jpy.cast(raster_Band.getGeoCoding(), GeoCoding)    #get geocoding of biomass map
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
coordsL = []

result = np.zeros(bioW * bioH, np.float32)
result.shape = bioH, bioW

for y in range (0, bio_data.shape[0]):  #shape[0] = 401 (height)
    for x in range (0, bio_data.shape[1]):
        if mask[y][x] == 1 and bio_data[y][x] >= 0:
            result[y][x] = bio_data[y][x]
            if bio_data[y][x] >= float(350):
                coordsH.append([x, y]) 
            elif bio_data[y][x] >= float(0):
                coordsL.append([x, y])         
        else:
            result[y][x] = -9999

imgplot = plt.imshow(result)
imgplot.write_png(productName + 'Result.png')

# Get pixel data from radar image using (x,y) coordinates
labels = []
pixelData = []
raster_BandRadar.loadRasterData()

for x in range (0, len(coordsH)):
    if not (coordsH[x][0] >= radarW) or (coordsH[x][1] >= radarH):            #Checks if coordinates is within the range of the target
        pix = raster_BandRadar.getPixelFloat(int(coordsH[x][0]), int(coordsH[x][1]))
        if (pix > 0):
                pixelData.append(pix)
                labels.append(2)

#for x in range (0, len(coordsL)):
#    if not (coordsL[x][0] >= radarW) or (coordsL[x][1] >= radarH):
#        pix = raster_BandRadar.getPixelFloat(int(coordsL[x][0]), int(coordsL[x][1]))
#        if (pix > 0):
#                pixelData.append(pix)
#                labels.append(0)

print(labels)
# Save data and labels
pickle.dump(pixelData, open(savePath + productName + '_' + bandName + '_newcollocateddata', 'wb'))
pickle.dump(labels, open(savePath + productName + '_' + bandName + '_newcollocatedlabels', 'wb'))
print("Data saved")

# Cleans up products
p.dispose()
r.dispose()