import csv
from pathlib import Path

EXP = "v3_long"
BASE_DIR =  Path("/home/ptrchv/repos/WifiRoaming/analysis")
TRAJ_DIR = BASE_DIR / "trajectories"/ EXP

TRAJ_FILE = TRAJ_DIR / "trajectory_optimized_10.csv"

with open(TRAJ_FILE, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    lines = []
    for row in reader:
        lines.append(row)

for idx in range(1, len(lines)):
    row = lines[idx]
    prev = lines[idx-1]
    if row["state"] == "ROAMING" and prev["state"] == "ROAMING" and row["ap"] != prev["ap"]:
        print("Problem found at line {}".format(idx))
