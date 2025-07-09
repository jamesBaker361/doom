for scenario in ["AquaticRuinZone.Act1",
                  "CasinoNightZone.Act1",
                    "ChemicalPlantZone.Act1",
                    "EmeraldHillZone.Act1",
                    "HillTopZone.Act1",
                    "MetropolisZone.Act1"]:
    command=f"sbatch -J sonic --err=slurm_chip/sonic/{scenario}.err --out=slurm_chip/sonic/{scenario}.out runpygpu_chip.sh retrovideo.py --scenario {scenario} --timesteps 1000000 "
    print(command)
    