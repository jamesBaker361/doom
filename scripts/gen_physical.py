for image_encoder in ["vae","trained"]:
    for lr in [0.001, 0.005]:
        command=f"sbatch -J phy --err=slurm_chip/physical/{image_encoder}_{lr}.err --out=slurm_chip/physical/{image_encoder}_{lr}.out "
        command+=f"runpygpu_chip.sh physical_models.py --image_encoder {image_encoder} --epochs 100 --val_interval 10 --gradient_accumulation_steps 8 --limit 10000 "
        command+=f" --name jlbaker361/{image_encoder}_{lr} "
        print(command)