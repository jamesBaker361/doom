for model_type in ["concattransformer","concatlstm","concatgru","concatrnn","rnn","lstm","gru","transformer"]:
    command=f"sbatch -J seq --err=slurm/seq/{model_type}.err --out=slurm/seq/{model_type}.out runpygpu.sh "
    command+=f" sequence_training.py --model_type {model_type} --metadata_keys x y  --sequence_dataset  --sequence_dataset jlbaker361/sonic_sequence_100  "
    command+=f" --gradient_accumulation_steps 2 --batch_size 2 "
    if model_type.find("concat")!=-1:
        command+=" --use_prior  "
    print(command)