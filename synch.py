import os

def get_synched_frames(frameList, frameListRgb, frameListAccel):

    accelRecs = []
    scenes = os.listdir("data")
    scenes.sort()

    previousDepth = 0
    previousRgb = 0
    
    rgbImgs=[]
    depthImgs=[]
    
    for scene in scenes: 
        files = os.listdir("data/"+scene)
        files.sort()
        newFiles = []
        numDepth = 0   # number of frame depth
        numRgb = 0     # number of frame rgb
        numAccel = 0
        

        for file in files:
            sz = os.path.getsize("data/"+scene+"/"+file)
            if ( sz == 614417 or sz == 921615 ): #filter out the corrupted frames
                newFiles.append(file)
        
        files=newFiles 
        
        for i in range(len(files)):    
            if (files[i][0:1] == 'd'):
                numDepth += 1
                depthImgs.append(scene+"/"+files[i])
            elif (files[i][0:1] == 'r'):
                numRgb += 1
                rgbImgs.append(scene+"/"+files[i])                
        
        rgb_pointer = 0
        accel_pointer = 0

        for depth_pointer in range(numDepth):
            starting_idx = 3 + len(scene)
            ending_idx = 20 + len(scene)
            dp = depth_pointer + previousDepth
            timestampDepth = depthImgs[dp][starting_idx:ending_idx]
                                    
            if ( rgb_pointer+previousRgb < len(rgbImgs) ):
                timestampRgb = rgbImgs[rgb_pointer+previousRgb][starting_idx:ending_idx]

                tDepth = float(timestampDepth)
                tRgb = float(timestampRgb)

                tDiff = abs(tDepth-tRgb)

                while (rgb_pointer+1 < numRgb):
                    rp = rgb_pointer + previousRgb
                    nexttimestampRgb = rgbImgs[rp][starting_idx:ending_idx]
                    tRgb = float(nexttimestampRgb)

                    tmpDiff = abs(tDepth-tRgb)
                    if (tmpDiff > tDiff):
                        break
                        
                    tDiff=tmpDiff
                    rgb_pointer += 1
                
                frameList.append(depthImgs[depth_pointer+previousDepth])
                frameListRgb.append(rgbImgs[rgb_pointer+previousRgb])

        previousDepth = previousDepth + numDepth
        previousRgb = previousRgb + numRgb
