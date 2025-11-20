images=[]
import PIL
from datasets import load_dataset
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

data=load_dataset("jlbaker361/discrete_HillTopZone.Act11000000",split="train")

images=[]
duration=500
for n,row in enumerate(data):
    image=row["image"]
    draw=ImageDraw.Draw(image)
    text=str(row["action_combo"])
    font = ImageFont.load_default(20)
    draw.text((150,10),text,(255,255,255),font=font)
    images.append(image)
    
    

images[0].save('pillow_imagedraw.gif',
               save_all = True, append_images = images[1:],
               optimize = False, duration = duration//4)