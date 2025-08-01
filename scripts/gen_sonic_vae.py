port=29650

base_path="sonic_videos_50000/SonicTheHedgehog2-Genesis/"
image_folder_path_list=[base_path+expanded_path for expanded_path in [
   # "AquaticRuinZone.Act1/constitution-ketoacidosis-nystan",
   # "CasinoNightZone.Act1/underbrush-constitution-plebeian",
   # "ChemicalPlantZone.Act1/accordion-cash-hopi",
    "EmeraldHillZone.Act1/phylloscopus-wrongness-steam",
   # "HillTopZone.Act1/bogbean-unau-phoeniculus",
   # "MetropolisZone.Act1/clasp-beeper-injuriousness"
]]
for gpus in [2]:
    for zone in ["MetropolisZone.Act11/uzbekistan-goldilocks-solicitor",
                 "CasinoNightZone.Act11/complexness-kanarese-oleander",
                 "AquaticRuinZone.Act11/cetrimide-bovidae-schooldays",
                 "ChemicalPlantZone.Act11/bowdlerisation-mishegoss-uncial ",
                 "HillTopZone.Act11/acanthocytosis-gasterophilus-patrioteer",
                 "EmeraldHillZone.Act11/constrictor-lorazepam-sphaerocarpales"]:
        
        zone_name=zone.split("/")[0]
        command=f"sbatch -J vae  --nodelist=g20-02,g20-03,g20-04,g20-05,g20-09,g20-12 --err=slurm_chip/vae/{gpus}_{zone_name}/sonic.err --out=slurm_chip/vae/{gpus}_{zone_name}/sonic.out "
        command+=" runpygpu_chip.sh "
        port+=1
        command+=f" vae.py --image_folder_paths "
        command+=f" {base_path}{zone} "
        command+=f" --batch_size 2 --gradient_accumulation_steps 16  --limit -1 --epochs 100 --image_interval 5 --name jlbaker361/sonic-vae{gpus}-{zone_name} "
        command+=" --project_name vae --skip_frac 0.9 --project_name sonic-vae "
        print(command)