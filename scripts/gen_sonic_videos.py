for num in [1,2]:
  for _scenario in ["AquaticRuinZone.Act",
                    "CasinoNightZone.Act",
                      "ChemicalPlantZone.Act",
                      "EmeraldHillZone.Act",
                      "HillTopZone.Act",
                      "MetropolisZone.Act"]:
      scenario=f"{_scenario}{num}"
      command=f"sbatch -J sonic --err=slurm_chip/sonic/{scenario}.err --out=slurm_chip/sonic/{scenario}.out runpygpu_chip.sh retrovideo.py --scenario {scenario} --timesteps 1000000 "
      print(command)
    