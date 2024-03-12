import os
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import box,MultiPolygon,Polygon
from rasterio.features import shapes
from shapely.geometry import shape
from rasterio.plot import show
from rasterio.features import geometry_mask
#import matplotlib.pyplot as plt
import shutil
import glob
import pandas as pd
import simplekml

import math
import openpyxl
from shapely.geometry import box,shape,mapping
from sklearn.metrics import mean_squared_error
import re

inputFolder = "F:/myenv/S2_input/" # S2 input rasters consisting of pre burn composite and post burn rasters
inputFileList=os.listdir(inputFolder)

def extractDate(filename):# function for extracting dates from the input files
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    return match.group() if match else ''

# Sort the filenames based on the extracted date
inputSortedList = sorted(inputFileList, key=extractDate)

#inputSortedList=sorted(inputFileList,key=lambda x:(not x.startswith('pre'),x)) #sorts the input folder such that the filename starting with 'pre' comes first

binaryOutputFolder = "F:/myenv/S2_binary/" # S2 binary output folder


burnOutputPath = os.path.join(binaryOutputFolder, f"burnmap_nbr.tif")# path where the changed burn map would be stored
cropMaskPath=r'F:\myenv\cropMask.tif' #path where cropmask tif is there

excelFilePath=r'F:\myenv\burnareas.xlsx'
groundTruthExcelPath=r'F:\myenv\groundtruth_nov2023.xlsx'
kmzPath=r'F:\myenv\burnmap.kmz'
shutil.rmtree(binaryOutputFolder)# deletes all previous files from the folder
if not os.path.exists(binaryOutputFolder):
    os.makedirs(binaryOutputFolder)
    
    
inputFiles = [f for f in inputSortedList if f.endswith('.tif')]
nbrChangeOutputPath = os.path.join(binaryOutputFolder, f"nbrChange.tif")# Create the output file path for nbr change
nbr2ChangeOutputPath = os.path.join(binaryOutputFolder, f"nbr2Change.tif")# Create the output file path for nbr2 change
mirbiChangeOutputPath = os.path.join(binaryOutputFolder, f"mirbiChange.tif")# Create the output file path for mirbi change
ndbiChangeOutputPath = os.path.join(binaryOutputFolder, f"ndbiChange.tif")# Create the output file path for ndbi change
baiS2ChangeOutputPath = os.path.join(binaryOutputFolder, f"baiChange.tif")# Create the output file path for ndbi change
afriChangeOutputPath = os.path.join(binaryOutputFolder, f"afriChange.tif")# Create the output file path for ndbi change
tbiChangeOutputPath = os.path.join(binaryOutputFolder, f"tbiChange.tif")# Create the output file path for tbi change
indicesChangeList=list()
indicesChangeList.extend([nbrChangeOutputPath,nbr2ChangeOutputPath,mirbiChangeOutputPath,ndbiChangeOutputPath,baiS2ChangeOutputPath,afriChangeOutputPath,tbiChangeOutputPath])

def spectralCalc(): # function for calculating the spectral indices
    spectralRastersList=list()

    for inputFile in inputFiles:
        #print('ip file',inputFile)       
        inputPath = os.path.join(inputFolder, inputFile)
       # inputBasename = os.path.splitext(os.path.basename(inputPath))[0]
         
        
        # Initialize indices here
        nbr = None
        nbr2=None
        mirbi=None
        ndbi=None
        bais2=None
        afri=None
        tbi=None

               
        # Open the input image with Rasterio
        with rasterio.open(inputPath) as src:
            # rasterData=src.read(1)
            # profile=src.profile

            # bbox = [74.764, 31.155, 75.478, 31.71]
            
           
            # #print('bboxlist',bboxList)        
            # #print('bbox',bboxList[0][1])  
            # boundingBoxGeometries=box(*bbox)
            # #print('boundgeom',len(bboxList))
            # multiPolygon=Polygon(boundingBoxGeometries)
            # mask=geometry_mask([multiPolygon],out_shape=src.shape,transform=src.transform,invert=True)  
            # rasterData[mask]=0 

            # rasterDataPath = os.path.join(inputFolder, f"{inputBasename}_clipped.tif")

            # with rasterio.open(rasterDataPath, 'w', **profile) as dst:
            #  dst.write(rasterData, 1)
            
                            
            # Read the bands
            
            nirBand = src.read(1)
           
            nirBand= np.where(np.isnan(nirBand) | (nirBand < 0), 0, nirBand) # assigning nodata values to zero

             
            redBand = src.read(2)
            redBand=np.where(np.isnan(redBand) | (redBand<0),0,redBand)# assigning nodata values to zero

            greenBand=src.read(3)
            greenBand=np.where(np.isnan(greenBand) | (greenBand<0),0,greenBand)# assigning nodata values to zero
                       
                       
            blueBand=src.read(4)
            blueBand=np.where(np.isnan(blueBand) | (blueBand<0),0,blueBand)# assigning nodata values to zero
                       

            redEdge1=src.read(6)
            redEdge1=np.where(np.isnan(redEdge1) | (redEdge1<0),0,redEdge1)# assigning nodata values to zero

            redEdge2=src.read(7)
            redEdge2=np.where(np.isnan(redEdge2) | (redEdge2<0),0,redEdge2)# assigning nodata values to zero

            redEdge3=src.read(8)
            redEdge3=np.where(np.isnan(redEdge3) | (redEdge3<0),0,redEdge3)# assigning nodata values to zero
           
           
            swir1Band = src.read(9)
            swir1Band=np.where(np.isnan(swir1Band) | (swir1Band<0),0,swir1Band) # assigning nodata values to zero
             
                                 
            swir2Band = src.read(10)
            swir2Band=np.where(np.isnan(swir2Band) | (swir2Band<0),0,swir2Band)# assigning nodata values to zero
           

                          

                       
            ################ CALCULATING NBR ##############################

            # Create a mask for zero values in both NIR and SWIR2 bands
            nbrZeroMask = (nirBand + swir2Band) == 0
                       
            # Replace zero values with a small non-zero value to avoid division by zero
            nbrDenomNoZeros = np.where(nbrZeroMask, 1e-8, (nirBand+swir2Band))
            nbr=(nirBand-swir2Band)/(nbrDenomNoZeros)
              
           
            spectralRastersList.append({'indexName':'NBR','indexArray':nbr})
            
                        
           ############################################################## 
                      
            
        

            ############################ CALCULATING NBR2#####################################
             
            
            nbr2ZeroMask = (swir1Band + swir2Band) == 0 # Create a mask for zero values in both SWIR1 and SWIR2 bands

            nbr2DenomNoZeros = np.where(nbr2ZeroMask, 1e-8, (swir1Band+swir2Band))# Replace zero values with a small non-zero value to avoid division by zero
            nbr2=(swir1Band-swir2Band)/(nbr2DenomNoZeros)# Only calculate NBR2 for non-zero values of denominator
            
              
            spectralRastersList.append({'indexName':'NBR2','indexArray':nbr2})
            #######################################################################################

             ################ CALCULATING MIRBI ##############################

            mirbi=10*swir2Band-9.8*swir1Band + 2
              
            
            spectralRastersList.append({'indexName':'MIRBI','indexArray':mirbi})
            
                        
           ##############################################################
            
            ############################ CALCULATING  CUSTOM NDBI #####################################
             
                
            caiDenomNoZeros = np.where(swir1Band==0, 1e-8, swir1Band)# Replace zero values with a small non-zero value to avoid division by zero
            cai=swir2Band/caiDenomNoZeros# Only calculate cai for non-zero values of denominator
            qNbrDenomNoZeros=np.where(swir2Band==0, 1e-8, swir2Band)# Replace zero values with a small non-zero value to avoid division by zero
            qNbr=nirBand/qNbrDenomNoZeros
            ndbiNoZeroMask=np.where((cai+qNbr==0),1e-8,cai+qNbr)

            ndbi=(cai-qNbr)/(ndbiNoZeroMask)
                        
                
            spectralRastersList.append({'indexName':'NDBI','indexArray':ndbi})
            #######################################################################################



            ############################# CALCULATING BURNT AREA INDEX2 BAIs2 ###############################################
            denomNoZeroRedBand=np.where(redBand==0,1e-8,redBand)
            denomNoZeroSw2Re3=np.where(swir2Band+redEdge3==0,1e-8,swir2Band+redEdge3)
            bais2 = (1-np.sqrt((redEdge1*redEdge2*redEdge3)/denomNoZeroRedBand))*(((swir2Band-redEdge3)/(np.sqrt(denomNoZeroSw2Re3)))+1)

            spectralRastersList.append({'indexName':'BAIS2','indexArray':bais2})
            #############################################################################################

            ############################# CALCULATING AEROSOL FREE INDEX(AFRI) ###############################################
             # Create a mask for zero values in both NIR and SWIR2 bands
            afriZeroMask = (nirBand + 0.5*swir2Band) == 0
                       
            # Replace zero values with a small non-zero value to avoid division by zero
            afriDenomNoZeros = np.where(afriZeroMask, 1e-8, (nirBand+0.5*swir2Band))
            afri=(nirBand-0.5*swir2Band)/(afriDenomNoZeros)
            spectralRastersList.append({'indexName':'AFRI','indexArray':afri})
            ##################################################################################################################

            ################### CALCULATING Tasseled-cap brightness(TBI)######################################
            tbi=0.3510*blueBand + 0.3813*greenBand + 0.3437*redBand+ 0.7196*nirBand + 0.2396*swir1Band+ 0.1949*swir2Band
            spectralRastersList.append({'indexName':'TBI','indexArray':tbi})
             ########################################################################################################     
                 
                
           
            
    return(spectralRastersList)

def changeDetection(spectralRastersList):# function for obtaining the burn map
    
    # Get a list of all raster files in the folder
    rasterFiles = [file for file in os.listdir(inputFolder) if file.endswith('.tif')]
    
     
    firstRasterPath = os.path.join(inputFolder, rasterFiles[0])# Open the first raster file(any file can be opened)

    with rasterio.open(firstRasterPath) as src:# Access raster data or metadata here
    
        data = src.read(1)  # Reads the first band as a NumPy array
        profile = src.profile  # Access metadata like CRS, transform, etc.
                

        netChangeSum=np.zeros_like(data,dtype=float)
        noOfIndices=len(spectralRastersList)//2
        indexChangeList=list() #list for storing the binary change maps of each index 
        
        ################################Computing binary change maps for each index###########################################################################
        for index in  range(0,noOfIndices):
            print('index',index,spectralRastersList[index]['indexName'])
            if (spectralRastersList[index]['indexName']=='NBR') or (spectralRastersList[index]['indexName']=='AFRI'): 
                denomNoZeroes=np.where((spectralRastersList[index]['indexArray']==0),1e-8,spectralRastersList[index]['indexArray'])
                
                netChange=((spectralRastersList[index]['indexArray'])-(spectralRastersList[index+noOfIndices]['indexArray']))/(np.sqrt((np.abs(denomNoZeroes))/1000))#  dNBR or dAfri
                with rasterio.open(cropMaskPath) as mask:
                    cropLands=mask.read(1)
                    netChange=np.where((cropLands>=0.10),netChange,0)# applying a mask for croplands with probability >=10%
                   
                max=np.max(netChange)
                min=np.min(netChange) 
                normalisedArray=(netChange-min)/(max-min)
                maxmask=normalisedArray[(normalisedArray>0) & (normalisedArray<1)] 
                minmask=normalisedArray[(normalisedArray>0) & (normalisedArray<1)] 
                        
                
                print('max',np.max(maxmask))
                print('min',np.min(minmask))                                               
                                                        
                IndexbinaryChange=np.where(normalisedArray>=0.7,1,0)
                indexChangeList.append(IndexbinaryChange) 
                with rasterio.open(indicesChangeList[index], 'w', **profile) as dst: 
                    dst.write(IndexbinaryChange, 1)     

            elif (spectralRastersList[index]['indexName']=='NBR2'):  
                netChange=(spectralRastersList[index]['indexArray'])-(spectralRastersList[index+noOfIndices]['indexArray'])
                with rasterio.open(cropMaskPath) as mask:
                    cropLands=mask.read(1)
                    netChange=np.where((cropLands>=0.10),netChange,0)# applying a mask for croplands with probability >=10%
                max=np.max(netChange)
                min=np.min(netChange) 
                print('max',max)
                print('min',min)                                               
                                                        
                IndexbinaryChange=np.where(netChange>=0.2,1,0)
                indexChangeList.append(IndexbinaryChange) 
                with rasterio.open(indicesChangeList[index], 'w', **profile) as dst: 
                    dst.write(IndexbinaryChange, 1)  

            elif (spectralRastersList[index]['indexName']=='BAIS2'):  
                netChange=(spectralRastersList[index]['indexArray'])-(spectralRastersList[index+noOfIndices]['indexArray'])
                with rasterio.open(cropMaskPath) as mask:
                    cropLands=mask.read(1)
                    netChange=np.where((cropLands>=0.10),netChange,0)# applying a mask for croplands with probability >=10%
                max=np.max(netChange)
                min=np.min(netChange) 
                print('max',max)
                print('min',min)                                               
                                                        
                IndexbinaryChange=np.where(netChange>=0.2,1,0)
                indexChangeList.append(IndexbinaryChange) 
                with rasterio.open(indicesChangeList[index], 'w', **profile) as dst: 
                    dst.write(IndexbinaryChange, 1)
            elif (spectralRastersList[index]['indexName']=='TBI'):
                netChange=(spectralRastersList[index]['indexArray'])-(spectralRastersList[index+noOfIndices]['indexArray'])
                with rasterio.open(cropMaskPath) as mask:
                    cropLands=mask.read(1)
                    netChange=np.where((cropLands>=0.10),netChange,0)# applying a mask for croplands with probability >=10%
                max=np.max(netChange)
                min=np.min(netChange) 
                print('max',max)
                print('min',min)                                               
                                                        
                IndexbinaryChange=np.where(netChange>=0.3,1,0)
                indexChangeList.append(IndexbinaryChange) 
                with rasterio.open(indicesChangeList[index], 'w', **profile) as dst: 
                    dst.write(IndexbinaryChange, 1)


           
            elif (spectralRastersList[index]['indexName']=='MIRBI'):
                    netChange=(spectralRastersList[index]['indexArray'])-(spectralRastersList[index+noOfIndices]['indexArray'])
                    with rasterio.open(cropMaskPath) as mask:
                        cropLands=mask.read(1)
                        netChange=np.where((cropLands>=0.10),netChange,0) # applying a mask for croplands with probability >=10%
                    
                    max=np.max(netChange)
                    min=np.min(netChange) 
                    normalisedArray=(netChange-min)/(max-min)
                    maxmask=normalisedArray[(normalisedArray>0) & (normalisedArray<1)] 
                    minmask=normalisedArray[(normalisedArray>0) & (normalisedArray<1)] 
                         
                 
                    print('max',np.max(maxmask))
                    print('min',np.min(minmask))  
                                                                               
                                                            
                    IndexbinaryChange=np.where(normalisedArray>=0.5,1,0)
                    indexChangeList.append(IndexbinaryChange) 
                    with rasterio.open(indicesChangeList[index], 'w', **profile) as dst: 
                        dst.write(IndexbinaryChange, 1) 
                    
                                        
            
                              
            else:
                    netChange=(3/4*spectralRastersList[index+noOfIndices]['indexArray'])-(1/4*spectralRastersList[index]['indexArray']) #Determining weighted temporal change for NDBI(as per the formula)
                    
                    print('max',np.max(netChange))
                    print('min',np.min(netChange))
                    
                    with rasterio.open(cropMaskPath) as mask:
                        cropLands=mask.read(1)
                        netChange=np.where((cropLands>=0.10),netChange,0)# applying a mask for croplands with probability >=10%
                    
                    IndexbinaryChange=np.where(netChange>=0.2,1,0)
                    indexChangeList.append(IndexbinaryChange)
                                            
                    with rasterio.open(indicesChangeList[index], 'w', **profile) as dst:
                        dst.write(IndexbinaryChange, 1) 

            ############################################################################################################################                   
                    
            netChangeSum+=netChange
            
        # stackedRasterArray=np.stack([indexChangeList[0],indexChangeList[1],indexChangeList[2],indexChangeList[3]],axis=2)
        # print('shape',stackedRasterArray.shape)  # stacking the binary change maps
        

        # #####################  Computing probability of burn at each pixel from the stacked arrays################################
        # burnMap=np.zeros_like(data,dtype=int) 
        # for row in  range (0,((stackedRasterArray.shape[0]))):
        #     count=0
        #     for col in range (0,((stackedRasterArray.shape[1]))):
        #         for array in range(0,((stackedRasterArray.shape[2]))):
        #             if(stackedRasterArray[row,col,array])==1:
        #                 count+=1
        #             else:
        #                 continue    
        #         probability=count/stackedRasterArray.shape[2] # calculating probability of burnt pixel
        #         if probability>=0.5: # if probability >= threshold 
        #             burnMap[row,col]=1
        #         else:
        #             burnMap[row,col]=0
        #         count=0    
            
        ###################################################################################################################

    #    # print(stackedRasterArray)   
        meanNetChange=netChangeSum/(len(spectralRastersList)/2)   

        # Convert changed burn map to a binary map
        changedBinaryMap = np.where(meanNetChange > 0.7, 1,0).astype(np.uint8)
        burnMap = np.where(meanNetChange == 0, 2, changedBinaryMap)#  assigning value 2 to 0 values in the array(they are zero due to no data values,unchanged regions,or non crop regions)
                
                
        with rasterio.open(burnOutputPath, 'w', **profile) as dst:
            dst.write(burnMap, 1)
        
        return burnMap  
    

def centroidAreaCalc():
   
   
    if os.path.isfile(excelFilePath) and excelFilePath.lower().endswith('.xlsx'):
        print('exel file exists')
        os.remove(excelFilePath)# remove the excel file already created
    else:
        print('excel file does not exist')
    workBook = openpyxl.Workbook()
    workBook.save(excelFilePath)#create the excel file again to store new values if required
    spectralRastersList = spectralCalc()# calling the function
    burntRasterMap = changeDetection(spectralRastersList)# calling the function
    
    shapesGen = shapes(burntRasterMap)
       
    latitudes = []
    longitudes = []
    areas = []
    

    earthRadius=6371000 #(in metres)
   
    # CALCULATE centroids of white areas in latitude and longitude and calculate area in hectares
    with rasterio.open(burnOutputPath) as ds:
    
        
        for geom, value in shapesGen:
            if value == 1:
                # Calculate area in sq degrees
                area = (shape(geom).area)*0.004046*2.47105 #area in acres
                
                   
                # Calculate centroid coordinates
                centroid = shape(geom).centroid
                lon, lat = ds.xy(centroid.x, centroid.y)
               
                                     

                # #Convert lat to radians
                # latRad=math.radians(lat)
                # #conversion factor for sq degrees to sq metres
                # #convFactor=(31.37 )*(math.pi/180)

                # #calculate area in hectares
                # areaMetres=area*(111000**2)*math.cos(lat) #area in sq metres
                # areaHectares=areaMetres*pow(10,-4) # area in hectares

                # # Append data to lists
                latitudes.append(lat)
                longitudes.append(lon)
                areas.append(area)
               
                
    # Create a DataFrame
    df = pd.DataFrame({'Latitude': latitudes, 'Longitude': longitudes, 'Area(acres)': areas})
        
    # Save DataFrame to Excel file
    df.to_excel(excelFilePath, index=False)
    # Create KML file from coordinates
    kml = simplekml.Kml()
    for index, row in df.iterrows():
        placeMark=kml.newpoint(name=f"P {index+1}", coords=[(row['Longitude'], row['Latitude'])])
        placeMark.description = f"Latitude: {row['Latitude']}\nLongitude: {row['Longitude']}\nArea: {row['Area(acres)']} hectares"

    # Save KML file to KMZ
    kml.savekmz(kmzPath)

centroidAreaCalc() # calling the function
print('Processing done')

    
    
    
    
    
        
            







                        
              
                   

                    
                       
            
                                  
                                       
            
            

           
            
            
           
   