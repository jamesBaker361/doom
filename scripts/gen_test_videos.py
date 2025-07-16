for name in ["castle","megaman","mario","sonic",
             "indiana","pink","mortal","street",
             "mortal2","mortal3"]:
    command=f"sbatch -J retro --err=slurm_chip/retro/{name}.err --out=slurm_chip/retro/{name}.out runpygpu_chip.sh retrovideo.py "
    game={
        "castle":"CastlevaniaBloodlines-Genesis",
        "megaman":"MegaManTheWilyWars-Genesis",
        "mario": "SuperMarioWorld-Snes",
        "sonic":"SonicTheHedgehog2-Genesis",
        "indiana":"IndianaJonesAndTheLastCrusade-Genesis",
        "pink":"PinkGoesToHollywood-Genesis",
        "mortal":"MortalKombatII-Genesis",
        "street":"StreetFighterIISpecialChampionEdition-Genesis",
        "mortal2":"MortalKombatII-Genesis",
        "mortal3":"MortalKombatII-Genesis",
    }[name]
    scenario={
        "sonic":"MetropolisZone.Act1",
        "megaman":"Level1.Swiv",
        "mario":"DonutPlains1",
        "castle":"Level1-2",
        "indiana":"Level1",
        "pink":"Level1",
        "mortal":"Level1.SubZeroVsRyden",
        "street":"Champion.Level1.RyuVsGuile",
        "mortal2":"Level1.JaxVsBaraka",
        "mortal3":"LiuKangVsKitana_VeryHard_05",
    }[name]
    command=f" sbatch -J test --err=slurm_chip/retro/{name}.err --out=slurm_chip/retro/{name}.out runpygpu_chip.sh retrovideo.py "
    command+=f" --scenario {scenario} --game {game} --timesteps 100000 --save_dir retro_diverse_videos --use_timelimit "
    print(command)