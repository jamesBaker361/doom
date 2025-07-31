for model_type in ["concattransformer","concatlstm","concatgru","concatrnn","rnn","lstm","gru","transformer"]:
    command=f"sbatch -J seq --err=slurm/seq/{model_type}.err --out=slurm/seq/{model_type}.out runpygpu.sh "
    command+=f" sequence_training.py --model_type {model_type} --metadata_keys x y --sequence_dataset  jlbaker361/sonic_hilltop_sequence --project_name sonic_hilltop"
    if model_type.find("concat")!=-1:
        command+=" --use_prior  "
    print(command)