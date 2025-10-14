for encoder_type in ["vae","vqvae"]:
    for grad in [4,8]:
        for lr in [0.001, 0.0001]:
            name=f"{encoder_type}_{grad}_{lr}"
            command=f" sbatch -J vae --err=slurm_chip/encoder/{name}.err --out=slurm_chip/encoder/{name}.out runpygpu_chip.sh vae.py "
            command+=f" --name  jlbaker361/sonic-encoder-{name} --save_dir sonic_encoder_{name}  --encoder_type  {encoder_type}  --project_name sonic-encoding --gradient_accumulation_steps {grad} "
            command+=f" --lr {lr} "
            print(command)