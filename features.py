import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image
import os

from shared import PATH,pil_grid


feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
 
# Parameters for Lucas Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
)

pil_data=Image.open(os.path.join(PATH,f"1.jpg"))

kernel=9
sigma=1

frame1= np.array(pil_data)[:, :, ::-1].copy()
#frame1=cv.GaussianBlur(frame1,(kernel,kernel),sigma)

old_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
h,w=old_gray.shape
mask=np.zeros_like(old_gray)
mask=cv.rectangle(mask, (70, 70), (135, 135), 1, -1)
masked_image=cv.bitwise_and(frame1,frame1,mask=mask)
pil_masked=Image.fromarray(cv.cvtColor(masked_image, cv.COLOR_BGR2RGB))
pil_masked.save("mask.jpg")

p0 = cv.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(frame1)

start=100
end=200
step=1
color = np.random.randint(0, 255, (100, 3))
buffer=10

new_path="videos_features"
os.makedirs(new_path,exist_ok=True)

def concat_unique_limit(a:np.ndarray, b, epsilon=1e-5):
    max_len=a.shape[0]+b.shape[0]
    c = np.concatenate([a, b], axis=0)
    unique_points = []

    for pt in c.reshape(-1,2):
        if not any(np.linalg.norm(pt - np.array(up)) < epsilon for up in unique_points):
            unique_points.append(pt)

    unique_points = np.array(unique_points).reshape(-1,1,2)
    if len(unique_points) > max_len:
        unique_points = unique_points[:max_len]

    return unique_points

for n in range(start,end,step):

    pil_data2=Image.open(os.path.join(PATH,f"{n}.jpg"))
    frame2=np.array(pil_data2)[:, :, ::-1].copy()
    #frame2=cv.GaussianBlur(frame2,(kernel,kernel),sigma)

    frame_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
 
    # Calculate Optical Flow
    try:
        p1, st, err = cv.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
    except Exception as err:
        print(err,n)
        break
    # Select good points
    try:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    except TypeError as err:
        print(err,n)
        break
 
    # Draw the tracks
    min_h=h
    min_w=w
    max_h=0
    max_w=0
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a,b,c,d=[int(x) for x in [a,b,c,d]]
        min_h=min(min_h,a)
        min_w=min(min_w,b)
        max_h=max(max_h,a)
        max_w=max(max_w,b)
    frame2=cv.rectangle(frame2, (min_h-buffer,min_w-buffer),(max_h+buffer,max_w+buffer),color=1,thickness=2)
        #mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        #frame2 = cv.circle(frame2, (a, b), 5, color[i].tolist(), -1)
    
 
    # Update the previous frame and previous points
    old_gray = frame_gray.copy()
    new_mask=np.zeros_like(old_gray)
    new_mask=cv.rectangle(new_mask, (min_h-buffer,min_w-buffer),(max_h+buffer,max_w+buffer), 1, -1)
    try:
        _p0 =  cv.goodFeaturesToTrack(old_gray, mask=new_mask, **feature_params)
        p0=concat_unique_limit(_p0,p0)
    except Exception as err:
        print(err,n)
        break
    frame2=cv.add(frame2,mask)
    pil_image = Image.fromarray(cv.cvtColor(frame2, cv.COLOR_BGR2RGB))
    merged=pil_grid([pil_image,pil_data2])
    merged.save(os.path.join(new_path,f"{n}.jpg"))

images=[Image.open(os.path.join(new_path,f"{n}.jpg")) for n in range(start,end,step) if os.path.exists(os.path.join(new_path,f"{n}.jpg"))]
images[0].save('sonic_features.gif',
               save_all = True, append_images = images[1:],
               optimize = True, duration = len([n in range(start,end,step)])//10)