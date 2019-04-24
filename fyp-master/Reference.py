#Subset image using geometry from wkt. But crop is square
wkt = "POLYGON ((-50.572411943319295 4.741338729858398, -50.88658239124524 3.229201555252075, -53.18025575713278 3.710718870162964, -52.87145928642018 5.218433380126953, -50.572411943319295 4.741338729858398))" 						
#test = "POLYGON ((" 
#test += str(topLeftLon) + " " + str(topLeftLat) + ", " + str(botLeftLon) + " " + str(botLeftLat) + ", " + str(botRightLon) + " " + str(botRightLat) + ", " + str(topRightLon) + " " + str(topRightLat) + ", " + str(topLeftLon) + " " + str(topLeftLat) + "))"
try:
    geometry = WKTReader().read(wkt)
    print(geometry)
except Exception as e:
    geometry = None
    print('Failed to convert WKT into geometry')
    print(e)
    
op = SubsetOp()
op.setSourceProduct(productTwo)
op.setGeoRegion(geometry)
sub_product = op.getTargetProduct()
ProductIO.writeProduct(sub_product, "D:/Test/testing.dim", "BEAM-DIMAP")

#Print pixeldata
raster_Band.loadRasterData()
pixelData = raster_band.getPixelFloat(10000,10000)
print("pixeldata: ", pixelData)

#Plot data using imgplot
#b1_data.shape = h, w
#r1_data.shape = h2, w2
#imgplot = plt.imshow(b1_data)
#imgplot = plt.imshow(r1_data)
#imgplot.write_png('band1.png')
#imgplot.write_png('intensity.png')

# jpy conversion
PixelPos = jpy.get_type('org.esa.snap.core.datamodel.PixelPos')
GeoPos = jpy.get_type('org.esa.snap.core.datamodel.GeoPos')
GeoCoding = jpy.get_type('org.esa.snap.core.datamodel.GeoCoding')
RasterDataNode = jpy.get_type('org.esa.snap.core.datamodel.RasterDataNode')
SubsetOp = jpy.get_type('org.esa.snap.core.gpf.common.SubsetOp')
    
# save training model
pickle.dump(model, open(filename, 'wb'))

# load training model
pickle.load(open(path, 'rb'))