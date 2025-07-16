base_path="sonic_videos_100000/SonicTheHedgehog2-Genesis/"
for [name,path] in [
    ["sonic_emerald_100000","EmeraldHillZone.Act1/aminoaciduria-popery-clozapine"],
    ["sonic_hilltop_100000","HillTopZone.Act1/quamassia-cervus-plateful"],
    ["sonic_metro_100000","MetropolisZone.Act1/sonograph-tinner-laudability"]
]:
    command=f"sbatch -J data --err=slurm_chip/data/{name}.err --out=slurm_chip/data/{name}.out runpygpu_chip.sh make_dataset.py "
    command+=f" --upload_path jlbaker361/{name} --folder {base_path}{path} "
    print(command)