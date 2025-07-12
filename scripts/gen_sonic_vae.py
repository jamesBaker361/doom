base_path="saved_retro_videos/SonicTheHedgehog2-Genesis/"
image_folder_path_list=[base_path+expanded_path for expanded_path in [
    "AquaticRuinZone.Act1/constitution-ketoacidosis-nystan",
    "CasinoNightZone.Act1/underbrush-constitution-plebeian",
    "ChemicalPlantZone.Act1/accordion-cash-hopi",
    "EmeraldHillZone.Act1/phylloscopus-wrongness-steam",
    "HillTopZone.Act1/bogbean-unau-phoeniculus",
    "MetropolisZone.Act1/clasp-beeper-injuriousness"
]]

command=f"sbatch -J vae --err=slurm_chip/vae/sonic.err --out=slurm_chip/vae/sonic.out runpygpu_chip.sh vae.py --image_folder_paths "
command+=" ".join(image_folder_path_list)
command+=f" --batch_size 2 --gradient_accumulation_steps 16  --limit -1 --epochs 20 --image_interval 5 --name jlbaker361/sonic-vae1.0 "
print(command)