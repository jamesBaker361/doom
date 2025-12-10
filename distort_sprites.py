from shared import game_window_dict,crop_black,crop_black_gray
import numpy as np
import os
import cv2 as cv

for game,shape in game_window_dict.items():
    canvas_h,canvas_w=shape
    directory=os.path.join("sprite_from_sheet",game)
    new_directory=os.path.join("distorted_sprite_from_sheet",game)
    os.makedirs(new_directory,exist_ok=True)
    for file in os.listdir(directory):
        if file.endswith("jpg"):
            
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            img=cv.imread(os.path.join(directory,file))

            # Get image size
            h, w = img.shape[:2]
            

            # Compute top-left coordinates to center the image
            x_offset = (canvas_w - w) // 2
            y_offset = (canvas_h - h) // 2
            try:
                # Paste the image on the canvas
                canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img

                # Optional: resize the whole canvas to another size
                resized_canvas = cv.resize(canvas, (256, 256))

                resized_canvas=crop_black(resized_canvas)
                new_path=os.path.join(new_directory,file)
                cv.imwrite(new_path,resized_canvas)
            except ValueError:
                print(os.path.join(directory,file),h,w)