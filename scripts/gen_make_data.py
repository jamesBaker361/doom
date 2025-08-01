for n in [50000]:
    base_path=f"sonic_videos_{n}/SonicTheHedgehog2-Genesis/"
    for [name,path] in [
    # ["sonic_emerald_100000_pretrained","EmeraldHillZone.Act1/aminoaciduria-popery-clozapine","jlbaker361/sonic-vae2-EmeraldHillZone.Act1"],
    #  ["sonic_hilltop_100000_pretrained","HillTopZone.Act1/quamassia-cervus-plateful","jlbaker361/sonic-vae2-HillTopZone.Act1"],
        [f"sonic_metro_{n}","MetropolisZone.Act11/uzbekistan-goldilocks-solicitor"],
        [f"sonic_casino_{n}","CasinoNightZone.Act11/complexness-kanarese-oleander"],
        [f"sonic_aqua_{n}","AquaticRuinZone.Act11/cetrimide-bovidae-schooldays"],
        [f"sonic_chemical_{n}","ChemicalPlantZone.Act11/bowdlerisation-mishegoss-uncial"],
        [f"sonic_hilltop_{n}","HillTopZone.Act11/acanthocytosis-gasterophilus-patrioteer"],
        [f"sonic_emerald_{n}","EmeraldHillZone.Act11/constrictor-lorazepam-sphaerocarpales"]
    ]:
        command=f"sbatch -J data  --nodelist=g20-02,g20-03,g20-04,g20-05,g20-09,g20-12 --err=slurm_chip/data/{name}.err --out=slurm_chip/data/{name}.out runpygpu_chip.sh make_dataset.py "
        command+=f" --upload_path jlbaker361/{name} --folder {base_path}{path}  "
        print(command)


'''

for zone in ["MetropolisZone.Act11/uzbekistan-goldilocks-solicitor",
                  "CasinoNightZone.Act11/complexness-kanarese-oleander",
                  "AquaticRuinZone.Act11/cetrimide-bovidae-schooldays",
                  "ChemicalPlantZone.Act11/bowdlerisation-mishegoss-uncial ",
                  "HillTopZone.Act11/acanthocytosis-gasterophilus-patrioteer",
                  "EmeraldHillZone.Act11/constrictor-lorazepam-sphaerocarpales"]:
'''