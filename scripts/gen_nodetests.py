def to_two_digit(match):
    return f"{int(match.group()):02d}"

for node in [f"g20-{to_two_digit(n)}" for n in range(1,14)]+[f"g24-{to_two_digit(1,13)}"]:
    print(f"sbatch --nodelist={node} runpygpu_chip.sh bullshit.py ")