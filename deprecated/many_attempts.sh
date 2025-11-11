
sbatch --constraint='RTX_2080TI' runpygpu_chip.sh rom_import.py
sbatch --constraint='RTX_6000' runpygpu_chip.sh rom_import.py
sbatch --constraint='RTX_8000' runpygpu_chip.sh rom_import.py
sbatch --constraint='H100' runpygpu_chip.sh rom_import.py
sbatch --constraint='L40S' runpygpu_chip.sh rom_import.py
