for timesteps in [2_000_000]:
  for num in [1]:
    for state in [#"AquaticRuinZone.Act1",
                     # "CasinoNightZone.Act1",
                      #  "ChemicalPlantZone.Act1",
                       "EmeraldHillZone.Act",]:
        scenario=f"{state}{num}"
        command=f"sbatch -J sonic --err=slurm/sonic/{scenario}{timesteps}.err --out=slurm/sonic/{scenario}{timesteps}.out runpymain.sh retrovideo.py --scenario {scenario} --timesteps {timesteps} "
        command+=f" --save_dir /scratch/jlb638/retro/sonic_videos_{timesteps} --state {scenario} "
        print(command)
    