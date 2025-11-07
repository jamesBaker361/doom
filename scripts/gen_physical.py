for image_encoder in ["vae","trained"]:
    command=f"sbatch -J --err=slurm_chip/physical/{image_encoder}.err --out=slurm_chip/physical/{image_encoder}.out "
    command+=f"runpygpu_chip.sh physical_models.py --image_encoder {image_encoder}"
    print(command)