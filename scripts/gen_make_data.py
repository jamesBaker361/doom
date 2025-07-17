base_path="sonic_videos_100000/SonicTheHedgehog2-Genesis/"
for [name,path,pretrained] in [
    ["sonic_emerald_100000_trained","EmeraldHillZone.Act1/aminoaciduria-popery-clozapine","jlbaker361/sonic-vae2-EmeraldHillZone.Act1"],
    ["sonic_hilltop_100000_trained","HillTopZone.Act1/quamassia-cervus-plateful","jlbaker361/sonic-vae2-HillTopZone.Act1"],
    ["sonic_metro_100000_trained","MetropolisZone.Act1/sonograph-tinner-laudability","jlbaker361/sonic-vae2-MetropolisZone.Act1"]
]:
    command=f"sbatch -J data --err=slurm_chip/data/{name}.err --out=slurm_chip/data/{name}.out runpygpu_chip.sh make_dataset.py "
    command+=f" --upload_path jlbaker361/{name} --folder {base_path}{path} "
    print(command)