for name in ["castle","megaman","mario"]:
    command=f"sbatch -J retro --err=slurm_chip/retro/{name}.err --out=slurm_chip/retro/{name}.out runpygpu_chip.sh retrovideo.py "
    game={
        "castle":"CastlevaniaBloodlines-Genesis",
        "megaman":"MegaManTheWilyWars-Genesis",
        "mario": "SuperMarioWorld-Snes",
        "sonic":"SonicTheHedgehog2-Genesis",
        "indiana":"IndianaJonesAndTheLastCrusade-Genesis",
        "pink":"PinkGoesToHollywood-Genesis",
        "mortal":"MortalKombatII-Genesis",
        "street":"StreetFighterIISpecialChampionEdition-Genesis"
    }[name]
    scenario={
        "sonic":"MetropolisZone.Act1",
        "megaman":"Level1.Swiv",
        "mario":"DonutPlains1",
        "castle":"Level1-2",
        "indiana":"Level1",
        "pink":"Level1",
        "mortal":"Level1.SubZeroVsRyden",
        "street":"Champion.Level1.RyuVsGuile"
    }[name]
    command=f" sbatch -J test --err=slurm_chip/test/{name}.err --out=slurm_chip/test/{name}.out runpygpu_chip.sh retrovideo.py "
    command+=f" --scenario {scenario} --game {game} --timesteps 1 "
    print(command)