def to_two_digit(match):
    if match>9:
        return str(match)
    else:
        return "0"+str(match)

for node in [f"g20-{to_two_digit(n)}" for n in range(1,14)]+[f"g24-{to_two_digit(1,13)}"]:
    print(f"sbatch --nodelist={node} runpygpu_chip.sh bullshit.py ")