for game in ["MarioBros-v5","BasicMath-v5","Centipede-v5","Crossbow-v5",
             "DonkeyKong-v5","ElevatorAction-v5","Frogger-v5","Gopher-v5",
             "HauntedHouse-v5","Jamesbond-v5","Kangaroo-v5","PrivateEye-v5"]:
    for mode in [0,1]:
        name=f"{game}_{mode}"
        command=f"sbatch -J atari --err=slurm_chip/atari/{name}.err --out=slurm_chip/atari/{name}.out "
        command+=f" runpygpu_chip.sh atarivideo.py --game {game} --mode {mode} --timesteps 500000"
        print(command)