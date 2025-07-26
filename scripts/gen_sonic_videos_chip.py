for timesteps in [2_500_000]:
  for num in [1]:
    for state in ["AquaticRuinZone.Act1",
                      "CasinoNightZone.Act1",
                        "ChemicalPlantZone.Act1",
                        "EmeraldHillZone.Act1",
                        "HillTopZone.Act1",
                        "MetropolisZone.Act1"]:
        scenario=f"{state}{num}"
        command=f"sbatch -J sonic --err=slurm_chip/sonic/{scenario}{timesteps}.err --out=slurm_chip/sonic/{scenario}{timesteps}.out runpygpu_chip.sh retrovideo.py --scenario {scenario} --timesteps {timesteps} "
        command+=f" --save_dir sonic_videos_{timesteps} --state {state} "
        print(command)
    