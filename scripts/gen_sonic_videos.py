for timesteps in [10]:
  for num in [1]:
    for _scenario in ["AquaticRuinZone.Act",
                      "CasinoNightZone.Act",
                        "ChemicalPlantZone.Act",
                        "EmeraldHillZone.Act",
                        "HillTopZone.Act",
                        "MetropolisZone.Act"]:
        scenario=f"{_scenario}{num}"
        command=f"sbatch -J sonic --err=slurm_chip/sonic/{scenario}{timesteps}.err --out=slurm_chip/sonic/{scenario}{timesteps}.out runpygpu_chip.sh retrovideo.py --scenario {scenario} --timesteps {timesteps} "
        command+=f" --save_dir sonic_videos_{timesteps} "
        print(command)
    