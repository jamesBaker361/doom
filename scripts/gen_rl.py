import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)

# Append the parent directory to sys.path
sys.path.append(parent_dir)
grandparent_dir=os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
from shared import game_state_dict


for game,values in game_state_dict.items():
    for state in values[:1]:
        name=f"{game}_{state}"
        command=f"sbatch -J rl --err=slurm_chip/rl/{name}.err --out=slurm_chip/rl/{name}.out runpygpu_chip.sh retrovideo_memory.py --game {game} --scenario {state} --hard_coded_steps 0 "
        command+=f" --timesteps 100 --repo_id jlbaker/{name}_agent  --dest_dataset  jlbaker/{name}_test_data --repo_id jlbaker/{name}_test_a "
        print(command)