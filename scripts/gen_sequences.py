for model_type in ["cnn","rnn","transformer","lstm","gru","concatrnn"]:
    command=f"sbatch -J seq --err=slurm/seq/{model_type}.err --out=slurm/seq/{model_type}.out runpymain.sh "
    command+=f" sequence_training.py --model_type {model_type} --metadata_keys x y  --sequence_dataset  jlbaker361/sonic_sequence_100 "
    print(command)