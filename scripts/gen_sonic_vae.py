port=29650

base_path="sonic_videos_100000/SonicTheHedgehog2-Genesis/"
image_folder_path_list=[base_path+expanded_path for expanded_path in [
   # "AquaticRuinZone.Act1/constitution-ketoacidosis-nystan",
   # "CasinoNightZone.Act1/underbrush-constitution-plebeian",
   # "ChemicalPlantZone.Act1/accordion-cash-hopi",
    "EmeraldHillZone.Act1/phylloscopus-wrongness-steam",
   # "HillTopZone.Act1/bogbean-unau-phoeniculus",
   # "MetropolisZone.Act1/clasp-beeper-injuriousness"
]]
for gpus in [2]:
    for zone in [
   # "AquaticRuinZone.Act1/constitution-ketoacidosis-nystan",
   # "CasinoNightZone.Act1/underbrush-constitution-plebeian",
   # "ChemicalPlantZone.Act1/accordion-cash-hopi",
    "EmeraldHillZone.Act1/aminoaciduria-popery-clozapine",
    "HillTopZone.Act1/quamassia-cervus-plateful",
    "MetropolisZone.Act1/sonograph-tinner-laudability"
   ]:
        
        zone_name=zone.split("/")[0]
        command=f"sbatch -J vae --err=slurm_chip/vae/{gpus}_{zone_name}/sonic.err --out=slurm_chip/vae/{gpus}_{zone_name}/sonic.out "
        command+=" runpygpu_chip.sh "
        port+=1
        command+=f" training/vae.py --image_folder_paths "
        command+=f" {base_path}{zone} "
        command+=f" --batch_size 2 --gradient_accumulation_steps 16  --limit -1 --epochs 20 --image_interval 5 --name jlbaker361/sonic-vae{gpus}-{zone_name} "
        command+=" --project_name vae --skip_frac 0.9"
        print(command)