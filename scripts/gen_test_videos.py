for name in ["castle","megaman","mario"]:
    command=f"sbatch -J retro --err=slurm_chip/retro/{name}.err --out=slurm_chip/retro/{name}.out runpygpu_chip.sh retrovideo.py "
    game={
        "castle":"CastlevaniaBloodlines-Genesis",
        "sonic":"SonicTheHedgehog2-Genesis",
        "mario": "SuperMarioWorld-Snes"
    }[name]
    scenario={
        "sonic":"MetropolisZone.Act1",
        "mario":"DonutPlains1",
        "castle":"Level1-2"
    }[name]
    command=f" sbatch -J test --err=slurm_chip/test/{name}.err --out=slurm_chip/test/{name}.out sbatch runpygpu_chip.sh retrovideo.py "
    command+=f" --scenario {scenario} --game {game} --timesteps 1 "
    print(command)