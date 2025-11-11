images=[]
import PIL
from datasets import load_dataset
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

data=load_dataset("jlbaker361/discrete_CasinoNightZone.Act110",split="train")

images=[]
duration=40
for n,row in enumerate(data):
    image=row["image"]
    draw=ImageDraw.Draw(image)
    draw.text((10,10),"Sample Text",(255,255,255))
    text=row["action_combo"]

images[0].save('pillow_imagedraw.gif',
               save_all = True, append_images = images[1:],
               optimize = False, duration = duration//4)