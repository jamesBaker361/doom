import retro
import pygame
import numpy as np
import os
import time

# ---- SETTINGS ----
RECORD_DIR = "records_user"
ACTIONS_FILE = "records_user/actions.npy"
os.makedirs(RECORD_DIR, exist_ok=True)

# Create environment with recording enabled
env = retro.make(
    game='SonicTheHedgehog2-Genesis',
    #state='GreenHillZone.Act1',
    record=RECORD_DIR,
    use_restricted_actions=retro.Actions.ALL
)

# ---- BUTTON MAPPING ----
# retro uses a fixed order of buttons depending on the game:
# Example Genesis mapping: ['B','NULL','C','A','Start','Up','Down','Left','Right']
BUTTONS = env.buttons

# Keyboard → Genesis button
KEY_TO_BUTTON = {
    pygame.K_RIGHT: "Right",
    pygame.K_LEFT: "Left",
    pygame.K_UP: "Up",
    pygame.K_DOWN: "Down",

    pygame.K_z: "B",      # Jump
    pygame.K_x: "A",
    pygame.K_c: "C",

    pygame.K_RETURN: "Start",
}

# Reverse lookup: button name → index
button_index = {b: i for i, b in enumerate(BUTTONS)}

# ---- Init pygame ----
pygame.init()
screen = pygame.display.set_mode((400, 200))
pygame.display.set_caption("Sonic Controller - Focus this window to play!")

clock = pygame.time.Clock()

obs = env.reset()
done = False

# Recorded list of full button vectors
action_log = []

print("Controls:")
print(" Arrow keys = Move")
print(" Z/X/C = Buttons (jump/spin)")
print(" Enter = Start")
print(" ESC = Quit")
print("------------------")

try:
    while not done:
        pressed = pygame.key.get_pressed()
        action = np.zeros(len(BUTTONS), dtype=np.int8)

        for key, button_name in KEY_TO_BUTTON.items():
            if pressed[key]:
                idx = button_index.get(button_name, None)
                if idx is not None:
                    action[idx] = 1

        # Per-step events (quit, ESC, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                done = True

        # Step emulator
        obs, rew, done, info = env.step(action)
        env.render()

        # Save user action
        action_log.append(action.copy())

        clock.tick(60)  # limit to 60 FPS for smooth control

except KeyboardInterrupt:
    pass

finally:
    env.close()
    pygame.quit()

    # Save the action sequence in numpy format
    np.save(ACTIONS_FILE, np.array(action_log, dtype=np.int8))
    print(f"Saved {len(action_log)} actions to {ACTIONS_FILE}")

    print("\nA .bk2 file has also been created in:", RECORD_DIR)
