for timesteps in [2_500_000]:
  for num in [1]:
    for ___scenario in [
                        "EmeraldHillZone.Act",]:
        scenario=f"{___scenario}{num}"
        command=f"sbatch -J sonic --err=slurm/sonic/{scenario}{timesteps}.err --out=slurm/sonic/{scenario}{timesteps}.out runpymain.sh retrovideo.py --scenario {scenario} --timesteps {timesteps} "
        command+=f" --save_dir /scratch/jlb638/retro/sonic_videos_{timesteps} "
        print(command)
    