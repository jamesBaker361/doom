import gymnasium as gym
import numpy as np


'''

SNES street fighter 2
A: Forward (medium kick)

B: Short (light kick)

X: Strong (medium punch)

Y: Jab (light punch)

L: Fierce (hard punch)

R: Roundhouse (hard kick)


Genesis Street fighter 2

A-	Light Kick
B-	 Medium Kick
C-	 Hard Kick

X- Light Punch
Y- Medium Punch
Z- Hard Punch

Genesis-SNES
B=A
A=B
C=R
X=Y
Y=X
Z=L

'''

# SNES: ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
SNES_BUTTONS = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]

# Genesis: ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
GENESIS_BUTTONS = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]

SHARED_ACTION_MAP ={
    'NOOP':[],
    "UP":['UP'],
    'DOWN':['DOWN'],
    'LEFT':['LEFT'],
    'RIGHT':['RIGHT'],
}

GENESIS_BUTTON_BINDING={
    'BA':["B"],
    'AB':['A'],
    'CR':['C'],
    'XY':['X'],
    'YX':['Y'],
    'ZL':['Z']
}

SNES_BUTTON_BINDING={
    'BA':["A"],
    'AB':['B'],
    'CR':['R'],
    'XY':['Y'],
    'YX':['X'],
    'ZL':['L']
}

SNES_MAP={}
GENESIS_MAP={}

for console_map,button_binding in zip([SNES_MAP,GENESIS_MAP],[SNES_BUTTON_BINDING,GENESIS_BUTTON_BINDING]):
    for shared_k,shared_v in SHARED_ACTION_MAP.items():
        for button_k,button_v in button_binding.items():
            console_map[f"{shared_k}-{button_k}"]=shared_v+button_v
        console_map[f"{shared_k}"]=shared_v

print("snes_map",SNES_MAP)
print("genesis_map",GENESIS_MAP)




def deets(console:str="Genesis"):
    if console=="Genesis":
        buttons=GENESIS_BUTTONS
        console_map=GENESIS_MAP
    elif console=="Snes":
        buttons=SNES_BUTTONS
        console_map=SNES_MAP
    _actions=[]
    for i,action in enumerate(console_map.keys()):
        button_combo = console_map[action]
        arr = np.array([False] * len(buttons))
        for btn in button_combo:
            arr[buttons.index(btn)] = True
        _actions.append(arr)
        print(i,action)
    
deets("Genesis")

print("klsdjjlfdslkfds")

deets("Snes")