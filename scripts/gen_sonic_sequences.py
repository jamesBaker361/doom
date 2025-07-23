for timesteps in [1_000_000]:
  for num in [1]:
    for _scenario in ["AquaticRuinZone.Act",
                      "CasinoNightZone.Act",
                        "ChemicalPlantZone.Act",
                        "EmeraldHillZone.Act",
                        "HillTopZone.Act",
                        "MetropolisZone.Act"]:
        scenario=f"{_scenario}{num}"
        command=f"sbatch -J sonic --err=slurm_chip/sonic_sequences/{scenario}{timesteps}.err --out=slurm_chip/sonic_sequences/{scenario}{timesteps}.out runpygpu_chip.sh retrovideo.py --scenario {scenario} --timesteps {timesteps} "
        command+=f" --save_dir sonic_sequences_{timesteps} --image_saving  --sequence_dataset jlbaker361/sonic_sequence_100   "
        print(command)
    