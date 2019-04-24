from snappy import ProductIO
from snappy import jpy
import numpy as np
import pickle

# Returns PixelPos object
def getPixCoord(geo_coding, lat, lon):
    pixPos = geo_coding.getPixelPos(GeoPos(float(lat), float(lon)), None)
    x = np.round(pixPos.getX())
    y = np.round(pixPos.getY())
    return (x, y)

# jpy conversion
PixelPos = jpy.get_type('org.esa.snap.core.datamodel.PixelPos')
GeoPos = jpy.get_type('org.esa.snap.core.datamodel.GeoPos')
GeoCoding = jpy.get_type('org.esa.snap.core.datamodel.GeoCoding')
RasterDataNode = jpy.get_type('org.esa.snap.core.datamodel.RasterDataNode')

# Preparing of product
productName = 'DF94'
fileName = 'D:/Test/AGB_2015_0.zip'
bandName = 'band_1'
savePath = 'D:/fyp-master/Polygon/'
polygon = []
topLeftLat = 4.431921373298707
topLeftLon = -52.14877215477987
botLeftLat = 2.885133324526054
botLeftLon = -52.468809464281705
botRightLat = 3.051240184812927
botRightLon = -53.258843412884204
topRightLat = 4.59619348038011
topRightLon = -52.9405767847531

radarImage = ProductIO.readProduct(fileName)
rasterBand = radarImage.getBand(bandName)

rasterBio = jpy.cast(rasterBand, RasterDataNode)
geo_CodingBio = jpy.cast(rasterBio.getGeoCoding(), GeoCoding) 

#Getting pixel positions
#POLYGON ((-54.459992631546 1.8573631048202515, -54.823678525590225 3.605360746383667, -52.60408776706431 4.070178031921387, -52.24492652190336 2.3271737098693848, -54.459992631546 1.8573631048202515))

pixPos = getPixCoord(geo_CodingBio, topLeftLat, topLeftLon)
print("Top Left: ", pixPos)
polygon.append(pixPos)

pixPos = getPixCoord(geo_CodingBio, botLeftLat, botLeftLon)
print("Bot Left: ", pixPos)
polygon.append(pixPos)

pixPos = getPixCoord(geo_CodingBio, botRightLat, botRightLon)
print("Bot Right: ", pixPos)
polygon.append(pixPos)

pixPos = getPixCoord(geo_CodingBio, topRightLat, topRightLon)
print("Top Right: ", pixPos)
polygon.append(pixPos)

#save polygon coordinates
pickle.dump(polygon, open(savePath + productName + 'Mask', 'wb'))


