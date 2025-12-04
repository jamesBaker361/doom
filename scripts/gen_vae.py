for encoder_type in ["vae"]:
    for grad in [16]:
        for lr in [0.0001,0.00001]:
            for zone in ["EmeraldHillZone","HillTopZone"]:
                name=f"{encoder_type}_{grad}_{lr}_{zone}"
                data=f"jlbaker361/discrete_{zone}.Act15000000"
                command=f" sbatch -J vae --constraint=L40S --err=slurm_chip/encoder/{name}.err --out=slurm_chip/encoder/{name}.out runpygpu_chip_l40.sh sonic_vae.py "
                command+=f" --repo_id jlbaker361/sonic-encoder-{name} --save_dir sonic_encoder_{name}  --encoder_type  {encoder_type}  --project_name sonic-encoding --gradient_accumulation_steps {grad} "
                command+=f" --lr {lr}  --epochs 250"
                command+=f" --process_data --skip_num 2 --src_dataset {data}  "
                print(command)