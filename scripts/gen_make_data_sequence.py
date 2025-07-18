base_path="sonic_sequences_1000000/SonicTheHedgehog2-Genesis/"
for [name,path] in [
    ["sonic_emerald_sequence","EmeraldHillZone.Act1/sinner-overlip-lightlessness"],
    ["sonic_hilltop_sequence","HillTopZone.Act1/tending-stripper-lingo"],
    ["sonic_metro_sequence","MetropolisZone.Act1/walbiri-ninepence-sphaerocarpales"]
]:
    command=f"sbatch -J data --err=slurm_chip/data/{name}.err --out=slurm_chip/data/{name}.out runpygpu_chip.sh make_dataset.py "
    command+=f" --upload_path jlbaker361/{name} --folder {base_path}{path} --no_image "
    print(command)