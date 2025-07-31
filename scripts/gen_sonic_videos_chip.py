for timesteps in [50000]:
  for num in [1]:
    for state in ["AquaticRuinZone.Act1",
                      "CasinoNightZone.Act1",
                        "ChemicalPlantZone.Act1",
                        "EmeraldHillZone.Act1",
                        "HillTopZone.Act1",
                        "MetropolisZone.Act1"]:
        scenario=f"{state}{num}"
        command=f"sbatch -J sonic --nodelist=g20-02,g20-03,g20-04,g20-05,g20-09,g20-12 --err=slurm_chip/sonic/{scenario}{timesteps}.err --out=slurm_chip/sonic/{scenario}{timesteps}.out runpygpu_chip.sh retrovideo.py --scenario {scenario} --timesteps {timesteps} "
        command+=f" --save_dir sonic_videos_{timesteps} --state {state} "
        print(command)
    