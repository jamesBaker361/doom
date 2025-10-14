for encoder_type in ["vae","vqvae"]:
    command=f" sbatch -J vae --err=slurm_chip/encoder/{encoder_type}.err --out=slurm_chip/encoder/{encoder_type}.out runpygpu_chip.sh vae.py "
    command+=f" --name  jlbaker361/sonic-encoder-{encoder_type} --save_dir sonic_encoder_{encoder_type}  --encoder_type  {encoder_type}  --project_name sonic-encoding --gradient_accumulation_steps 8 "