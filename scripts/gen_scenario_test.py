for timesteps in [10]:
    for ___scenario in ["AquaticRuinZone.Act",
                        "AquaticRuinZone.Act1",
                        "AquaticRuinZone",]:
        scenario=f"{___scenario}"
        command=f"sbatch -J sonic --err=slurm/sonic/{scenario}{timesteps}.err --out=slurm/sonic/{scenario}{timesteps}.out runpymain.sh retrovideo.py --scenario {scenario} --timesteps {timesteps} "
        command+=f" --save_dir /scratch/jlb638/retro/sonic_videos_{timesteps} --state {scenario} "
        print(command)
