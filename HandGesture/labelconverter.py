import csv
from PIL import Image
import numpy as np
import cv2
import os

IMG_DIR = '/home/itachi/Dset/hehe/'
ct2='/'
end=9
f = open("/home/itachi/Dset/hehe/combined.csv", "w")
writer = csv.writer(f)
for ct in range(1,end):
    for img in os.listdir(IMG_DIR+str(ct)+ct2):
        
            kt = Image.open(IMG_DIR+str(ct)+ct2+img).convert('L')
            kt = kt.resize((64, 64))
            #img_array = cv2.imread(os.path.join(IMG_DIR,img), cv2.IMREAD_GRAYSCALE)
            img_array = np.array(kt)
            img_array = (img_array.flatten())
    
            img_array  = img_array.reshape(-1, 1).T
    
            with open(IMG_DIR+str(ct)+ct2+'/output.csv', 'ab') as f:
    
                np.savetxt(f, img_array, delimiter=",")
    
    with open(IMG_DIR+str(ct)+ct2+'/output.csv','r') as csvinput:
        with open(IMG_DIR+str(ct)+ct2+'/'+str(ct)+'.csv', 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)
    
            all = []
    
            for row in reader:
                row.append(ct)
                for m in range(1,end):
                    if m==ct:
                        row.append(1)
                    else:
                        row.append(0)
                all.append(row)
    
            writer.writerows(all)

f = open(IMG_DIR+"combined.csv", "w")
writer = csv.writer(f)
for ct in range(1,end):
    reader = csv.reader(open(IMG_DIR+str(ct)+ct2+'/'+str(ct)+".csv"))
    for row in reader:
        writer.writerow(row)
        
for ct in range(1,end):
    os.remove(IMG_DIR+str(ct)+ct2+'/output.csv')
    os.remove(IMG_DIR+str(ct)+ct2+'/'+str(ct)+".csv")

