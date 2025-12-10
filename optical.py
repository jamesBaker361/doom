import numpy as np
import cv2 as cv
from PIL import Image
from shared import pil_grid
import os
import rembg
import os
import PIL
from PIL import Image
import cv2 as cv
from shared import pil_grid,crop_black
import numpy as np

eps=1
def get_nonzero_average(flow:np.ndarray):
    points=[]
    h,w=flow.shape[:2]
    for x in range(h):
        for y in range(w):
            gradient=flow[x][y]
            if abs(gradient[0])>eps or abs(gradient[1])>eps:
                points.append(gradient)
    if len(points)==0:
        return [0,0]
    return np.mean(np.array(points),axis=0)

def get_best(flow,mag,frac=0.025):
    rows=[]
    h,w=flow.shape[:2]
    for x in range(h):
        for y in range(w):
            rows.append([mag[x][y],flow[x,y]])
    rows.sort(key=lambda x: x[0],reverse=True)
    rows=[r[1] for r in rows]
    best=int(frac*len(rows))
    return np.mean(np.array(rows[:best]),axis=0)

def highpass_optical_flow(flow, blur_size=21, sigma=2):
    """
    flow: (H, W, 2) optical flow
    blur_size: kernel size for Gaussian (odd number)
    sigma: strength of smoothing
    
    returns high-pass filtered flow
    """
    # Smooth (low-frequency) version of flow
    flow_smooth = cv.GaussianBlur(flow, (blur_size, blur_size), sigma)

    # High-pass = original - smoothed
    flow_hp = flow - flow_smooth

    return flow_hp

BUFFER=5
END=15
SONIC_END=END

os.makedirs("box",exist_ok=True)
os.makedirs("sprite",exist_ok=True)
os.makedirs("cropped",exist_ok=True)

for game in ['SonicTheHedgehog2-Genesis','SuperMarioWorld-Snes','CastlevaniaBloodlines-Genesis']:
    for button in ["B","RIGHT","LEFT"]:
        PATH=os.path.join("videos_mini",game,button)


        pil_data=Image.open(os.path.join(PATH,f"1.jpg"))

        frame1=np.array(pil_data)[:, :, ::-1].copy()

        prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

        NEXT_PATH=os.path.join("videos_optical",game,button)
        os.makedirs(NEXT_PATH,exist_ok=True)

        start=0
        end=END
        step=1
        ''''        mask=np.zeros_like(prvs)
        min_h=70
        min_w=120
        max_h=130
        max_w=175
        mask=cv.rectangle(mask, (min_w,min_h), (max_w, max_h), 1, -1)
        masked_image=cv.bitwise_and(frame1,frame1,mask=mask)'''

        gray_list=[]
        bgr_list=[]
        frame_list=[]

        if game =='SonicTheHedgehog2-Genesis':
            end=SONIC_END

        for n in range(start,end,step):
            hsv = np.zeros_like(frame1)
            hsv[..., 1] = 255
            
            pil_data2=Image.open(os.path.join(PATH,f"{n}.jpg"))
            frame2=np.array(pil_data2)[:, :, ::-1].copy()
            frame_list.append(frame2.copy())

            next = cv.cvtColor(frame2.copy(), cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prvs, next, None,
                                                0.5,
                                                5, 
                                                15, 
                                                5, 
                                                7,
                                                    1.2, 0)
            #flow=highpass_optical_flow(flow) #adding this means it picks up on less
            '''grad=get_nonzero_average(flow[min_h:max_h,min_w:max_w])
            u=grad[0]
            v=grad[1]'''
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            mag_list=mag
            grad=[0,0] #get_best(flow[min_h:max_h,min_w:max_w],mag[min_h:max_h,min_w:max_w])
            u=grad[0]
            v=grad[1]
            ang=ang*180/np.pi/2
            hsv[..., 0] = ang
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

            gray=cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
            gray_list.append(gray.copy())
            bgr_list.append(bgr.copy())
            ''''mask_roi = np.zeros_like(prvs)
            mask_roi[min_h:max_h, min_w:max_w] = 255  # only this rectangle is considered

            # Apply mask to image (set other pixels to a dummy color, e.g., -1)
            masked_img = bgr.copy()
            masked_img[mask_roi == 0] = [-1, -1, -1]  # will ignore these'''
            #color_coverted = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
            '''grad_corner=flow[max_h,max_w]
            corner_u=-grad_corner[0]
            corner_v=-grad_corner[1]
            max_w=int(max_w+u+corner_u)
            min_w=int(min_w+u+corner_u)
            max_h=int(max_h+v+corner_v)
            min_h=int(min_h+v+corner_v)'''
            
            pil_image = Image.fromarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))
            #frame2=cv.rectangle(frame2,(min_w,min_h), (max_w, max_h),color=1,thickness=5)
            #pil_image = Image.fromarray(cv.cvtColor(frame2, cv.COLOR_BGR2RGB))
            merged=pil_grid([pil_image,pil_data2])
            merged.save(os.path.join(NEXT_PATH,f"{n}.jpg"))
            prvs=next
        '''average_mag=np.mean(mag_list)
        average_ang=np.mean(ang_list)
        average_flow=np.mean(flow_list)'''

        average_gray=np.mean(gray_list,axis=0)
        average_gray=cv.normalize(average_gray, None, 0, 255, cv.NORM_MINMAX)

        average_gray=cv.threshold(average_gray, 128, 255, cv.THRESH_BINARY)[1].astype(np.uint8)


        #Image.fromarray(cv.cvtColor(average_gray, cv.COLOR_GRAY2RGB)).save(f'gray_{game}_{button}.jpg')
        #mask=(average_gray > 0).astype(np.uint8)*255
        n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(average_gray, connectivity=8)

        best_label = None
        best_score = -np.inf


        # Loop over each component (skip 0 = background)
        for label in range(1, n_labels):
            component_mask = (labels == label)

            # Score = sum of magnitudes inside this region
            score = average_gray[component_mask].sum()

            if score > best_score:
                best_score = score
                best_label = label

        final_mask=labels==best_label
        final_mask=final_mask.astype(np.uint8)*255

        coords=np.column_stack(np.where(final_mask == 255))
        max_coord = coords.max(axis=0)
        min_coord = coords.min(axis=0)

        #print(max_coord,min_coord)

        (y1, x1) = min_coord
        y1-=BUFFER
        x1-=BUFFER
        (y2, x2) = max_coord
        y2+=BUFFER
        x2+=BUFFER

        final_image=cv.cvtColor(final_mask.copy(),cv.COLOR_GRAY2BGR)
        final_image=cv.rectangle(final_image, (x1, y1), (x2, y2),(0, 0, 255),2)
        Image.fromarray(cv.cvtColor(final_image, cv.COLOR_BGR2RGB)).save(os.path.join("box",f"{game}_{button}.jpg"))
        
        #gray_image = Image.fromarray(cv.cvtColor(final_mask, cv.COLOR_GRAY2RGB))
        #gray_image.save(f'components_{game}_{button}.jpg')

        pil_data2=Image.open(os.path.join(PATH,f"{1+n//2}.jpg"))
        frame2=np.array(pil_data2)[:, :, ::-1].copy()

        subset=frame2[y1:y2,x1:x2]
        #print(subset.shape,np.mean(subset))
        #print(minor,major)
        #sub_image=Image.fromarray(cv.cvtColor(subset, cv.COLOR_BGR2RGB))
        
        sub_image=Image.fromarray(cv.cvtColor(subset, cv.COLOR_BGR2RGB))
        sub_image.save(os.path.join("sprite",f"{game}_{button}.jpg"))
        #rem=rembg.remove(sub_image).convert("RGB")
        #rem.save(os.path.join("cropped",f"{game}_{button}.jpg"))

        #sprite=np.array(rem)[:, :, ::-1].copy()
        #sprite=crop_black(sprite)
        #Image.fromarray(cv.cvtColor(sprite, cv.COLOR_BGR2RGB)).save(os.path.join("sprite",f"{game}_{button}.jpg"))'''

        

        images=[Image.open(os.path.join(NEXT_PATH,f"{n}.jpg")) for n in range(start,end,step)]
        image_concat=final_image.copy()

        for gray,frame in zip(gray_list,frame_list):
            n_labels_gray, labels_gray, stats_gray, centroids_gray = cv.connectedComponentsWithStats(average_gray, connectivity=8)

            for label_gray in range(1, n_labels_gray):
                component_mask = (labels_gray == label_gray)

                # Score = sum of magnitudes inside this region
                score = gray[component_mask].sum()

                small_sum=gray[y1:y2,x1:x2].sum()

                if score>0 and small_sum/score >0.95:
                    break
            mask = component_mask.astype(np.uint8) * 255
            masked_bgr = cv.bitwise_and(frame,frame, mask=mask)
            image_concat=cv.hconcat([image_concat,masked_bgr])
        Image.fromarray(cv.cvtColor(image_concat, cv.COLOR_BGR2RGB)).save(f'panorama_{game}_{button}.jpg')
        print(f'{game}_{button}.gif')
        images[0].save(f'{game}_{button}.gif',
                    save_all = True, append_images = images[1:],
                    optimize = True, duration = len([n in range(start,end,step)])//4)