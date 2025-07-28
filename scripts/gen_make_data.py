base_path="sonic_videos_2500000/SonicTheHedgehog2-Genesis/"
for [name,path] in [
   # ["sonic_emerald_100000_pretrained","EmeraldHillZone.Act1/aminoaciduria-popery-clozapine","jlbaker361/sonic-vae2-EmeraldHillZone.Act1"],
  #  ["sonic_hilltop_100000_pretrained","HillTopZone.Act1/quamassia-cervus-plateful","jlbaker361/sonic-vae2-HillTopZone.Act1"],
    ["sonic_metro_2.5M","MetropolisZone.Act1/buttock-foramen-mariehamn"]
]:
    command=f"sbatch -J data --err=slurm_chip/data/{name}.err --out=slurm_chip/data/{name}.out runpygpu_chip.sh make_dataset.py "
    command+=f" --upload_path jlbaker361/{name} --folder {base_path}{path}  "
    print(command)