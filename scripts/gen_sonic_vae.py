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
    for zone in ["MetropolisZone.Act1/buttock-foramen-mariehamn",
                 "CasinoNightZone.Act11/scrapheap-carping-antivert",
                 "AquaticRuinZone.Act11/bogbean-spermatozoid-hercules",
                 "ChemicalPlantZone.Act11/computing-chinookan-foglamp",
                 "HillTopZone.Act11/cumfrey-scabbard-workbasket",
                 "EmeraldHillZone.Act11/chasuble-solicitor-ins"]:
        
        zone_name=zone.split("/")[0]
        command=f"sbatch -J vae --err=slurm_chip/vae/{gpus}_{zone_name}/sonic.err --out=slurm_chip/vae/{gpus}_{zone_name}/sonic.out "
        command+=" runpygpu_chip.sh "
        port+=1
        command+=f" vae.py --image_folder_paths "
        command+=f" {base_path}{zone} "
        command+=f" --batch_size 2 --gradient_accumulation_steps 16  --limit -1 --epochs 100 --image_interval 5 --name jlbaker361/sonic-vae{gpus}-{zone_name} "
        command+=" --project_name vae --skip_frac 0.975 --project_name sonic-vae "
        print(command)