for encoder_type in ["vae"]:
    for grad in [2]:
        for lr in [0.1,0.001,0.0001]:
            for zone in ["EmeraldHillZone"]: #,"HillTopZone"]:
                name=f"{encoder_type}_{grad}_{lr}_{zone}-load"
                data=f"jlbaker361/discrete_{zone}.Act1100"
                command=f" sbatch -J vae --constraint=L40S --err=slurm_chip/encoder/{name}.err --out=slurm_chip/encoder/{name}.out runpygpu_chip_l40.sh sonic_vae_load_test.py "
                command+=f" --name  jlbaker361/sonic-encoder-{name} --save_dir sonic_encoder_{name}  --encoder_type  {encoder_type}  --project_name sonic-encoding --gradient_accumulation_steps {grad} "
                command+=f" --lr {lr}  --epochs 10  "
                command+=f" --process_data --skip_num 2 --src_dataset {data} --limit 10 "
                print(command)