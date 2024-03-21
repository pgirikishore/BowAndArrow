#!/bin/bash
#SBATCH -J PythonJob               # Job name
#SBATCH -N 1                       # Number of nodes
#SBATCH -n 4                       # Number of tasks (assume your script can utilize 4 cores)
#SBATCH -o python_output_%j.txt    # Standard output file
#SBATCH -e python_error_%j.txt     # Standard error file

# Run your Python script
srun python BowAndArrowRL.py