from PIL import Image
import numpy as np
import cv2
NONE_STRING="None"
PATH="videos"
def pil_grid(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    return new_im

def crop_black(img:np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(np.max(gray),np.min(gray))
    coords = np.column_stack(np.where(gray > 10))  # any nonzero pixel

    if coords.size == 0:
        raise ValueError("Image is entirely black.")

    # Get bounding box
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = img[y0:y1+1, x0:x1+1]

    return cropped



def crop_black_gray(gray_img, threshold=10):
    # gray_img must be 2D: shape (H, W)
    if len(gray_img.shape) != 2:
        raise ValueError("Input must be a single-channel grayscale image.")

    # Find all pixels above threshold
    coords = np.column_stack(np.where(gray_img > threshold))

    if coords.size == 0:
        print("Image is entirely black.")
        return gray_img

    # Bounding box
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = gray_img[y0:y1+1, x0:x1+1]
    return cropped



SONIC_GAME='SonicTheHedgehog2-Genesis'
MARIO_GAME='SuperMarioWorld-Snes'
CASTLE_GAME='CastlevaniaBloodlines-Genesis'
SONIC_1GAME='SonicTheHedgehog-Genesis'

game_state_dict={
      #  SONIC_GAME:['EmeraldHillZone.Act1','AquaticRuinZone.Act1','CasinoNightZone.Act1','HillTopZone.Act1'],
        'SuperMarioWorld-Snes':["ChocolateIsland1",'DonutPlains1','Forest1','VanillaDome1'], #+["ChocolateIsland2",'DonutPlains2','Forest2','VanillaDome2'],
        'CastlevaniaBloodlines-Genesis':['Level1-1','Level2-1','Level3-1','Level4-1'], #+['Level1-2','Level2-2','Level3-2','Level4-2'],
        SONIC_1GAME:["GreenHillZone.Act1","LabyrinthZone.Act1","MarbleZone.Act1","ScrapBrainZone.Act1","SpringYardZone.Act1"], #+["GreenHillZone.Act2","LabyrinthZone.Act2","MarbleZone.Act2","ScrapBrainZone.Act2","SpringYardZone.Act2"]
    }

game_key_dict={
    'CastlevaniaBloodlines-Genesis':['gems', 'levelHi', 'lives', 'health', 'levelLo', 'map', 'score', 'action', 'save_path'],
    'SuperMarioWorld-Snes':['coins', 'score', 'lives', 'action', 'save_path'],
    SONIC_GAME:['act', 'game_mode', 'level_end_bonus', 'score', 'lives', 'rings', 'screen_x_end', 'screen_x', 'screen_y', 'x', 'y', 'zone', 'action', 'save_path'],
    SONIC_1GAME: ['act', 'level_end_bonus', 'score', 'lives', 'rings', 'screen_x_end', 'screen_x', 'screen_y', 'x', 'y', 'zone', 'action', 'save_path']
}
game_window_dict={
    SONIC_1GAME:[224, 320], #['EmeraldHillZone.Act1','AquaticRuinZone.Act1','CasinoNightZone.Act1','HillTopZone.Act1'],
        'SuperMarioWorld-Snes':[224, 256], #["ChocolateIsland1",'DonutPlains1','Forest1','VanillaDome1'],
        'CastlevaniaBloodlines-Genesis': [224, 320], #['Level1-1','Level2-1','Level3-1','Level4-1'],
        SONIC_GAME:[224,320]
}
all_states=[]
all_tokens=[]
all_games=[]
for key,value_list in game_state_dict.items():
    all_tokens.append(key)
    all_games.append(key)
    for v in value_list:
        all_tokens.append(v)
        all_states.append(v)
