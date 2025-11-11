for encoder_type in ["vae"]:
    for grad in [16]:
        for lr in [0.001, 0.0001]:
            name=f"{encoder_type}_{grad}_{lr}"
            command=f" sbatch -J vae --constraint=L40S --err=slurm_chip/encoder/{name}.err --out=slurm_chip/encoder/{name}.out runpygpu_chip_l40.sh sonic_vae.py "
            command+=f" --name  jlbaker361/sonic-encoder-{name} --save_dir sonic_encoder_{name}  --encoder_type  {encoder_type}  --project_name sonic-encoding --gradient_accumulation_steps {grad} "
            command+=f" --lr {lr} "
            print(command)