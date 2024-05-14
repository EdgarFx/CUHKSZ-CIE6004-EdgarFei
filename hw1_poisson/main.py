import os
import sys
import cv2
import numpy as np

## my function
import poisson

### 1. prepare config
argvs = sys.argv
src_path = argvs[1]
mask_path = argvs[2]
tar_path = argvs[3]
method = argvs[4]
src_name,src_extension = os.path.splitext(os.path.basename(src_path))
tar_name,tar_extension = os.path.splitext(os.path.basename(tar_path))
src_dir,src_detail_name = os.path.split(src_path)
tar_dir,tar_detail_name = os.path.split(tar_path)

### 2. for output
output_dir = "{0}/result".format(tar_dir)
if(not(os.path.exists(output_dir))):
  os.mkdir(output_dir)

outname = "{0}/result_{1}{2}".format(output_dir, method, tar_extension)
outname_cloning = "{0}/cloning{1}".format(output_dir, tar_extension)
outname_merged = "{0}/merged_result_{1}{2}".format(output_dir, method, tar_extension)
outfile = [outname,outname_cloning,outname_merged]

### 3. load images
src = np.array(cv2.imread(src_path,1)/255.0,dtype=np.float32)
tar = np.array(cv2.imread(tar_path,1)/255.0,dtype=np.float32)
mask = np.array(cv2.imread(mask_path,0),dtype=np.uint8)
ret,mask = cv2.threshold(mask,0,255,cv2.THRESH_OTSU)

### 4. apply poisson image editing
seamless_cloning, cloning = poisson.poissonImageEditing(src, mask/255.0, tar, method)
merged_result = np.hstack((np.array(src*255, dtype=np.uint8), cv2.merge((mask, mask, mask)), np.array(tar*255, dtype=np.uint8), cloning, seamless_cloning))

### 5. save result
cv2.imwrite(outname_merged, merged_result)
cv2.imwrite(outname_cloning, cloning)
cv2.imwrite(outname, seamless_cloning)
cv2.waitKey(0)
