for vae_checkpoint in ["default","none"]:
    for action in ["encoder","embedding"]:
        for grad in [16]:
            for lr in [0.0001,0.001]:
                for zone in ["EmeraldHillZone"]:
                    name=f"{action}_{lr}_{grad}_{vae_checkpoint}"
                    checkpoint={
                        "default":"jlbaker361/sonic-encoder-vae_16_1e-05_EmeraldHillZone"
                        ,"none":""
                    }[vae_checkpoint]
                    data=f"jlbaker361/discrete_{zone}.Act15000000" #jlbaker361/discrete_HillTopZone.Act15000000
                    command=f" sbatch -J render --constraint=L40S --err=slurm_chip/render/{name}.err --out=slurm_chip/render/{name}.out runpygpu_chip_l40.sh rendering_model.py "
                    command+=f" --repo_id  jlbaker361/rendering-{name} --save_dir rendering_{name}   --project_name sonic-rendering --gradient_accumulation_steps {grad} "
                    command+=f" --lr {lr}  --epochs 100"
                    command+=f"   --dataset {data} --vae_checkpoint {checkpoint} --use_lora"
                    print(command)