for model_type in ["cnn","rnn","transformer"]:
    command=f"sbatch -J seq --err=slurm_chip/seq/{model_type}.err --out=slurm_chip/seq/{model_type}.out runpygpu_chip.sh "
    command+=f" sequence_training.py --model_type {model_type} --metadata_keys x y "
    print(command)