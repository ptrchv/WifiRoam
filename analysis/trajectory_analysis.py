# %%
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict, fields

EXP = "v2_beacons"
BASE_DIR =  Path("/home/ptrchv/repos/WifiRoaming/analysis")
TRAJ_DIR = BASE_DIR / "trajectories"/ EXP


@dataclass
class Result:
    plr: float = None
    lat_mean: float = None
    lat_99: float = None
    lat_99_9: float = None
    num_tries_mean: float = None
    num_tries_99: float = None
    num_tries_99_9: float = None
    rssi_mean: float = None
    rssi_99: float = None
    rssi_99_9: float = None
    pkt_roaming: int = None
    pkt_disconnected: int = None

# %%
df_dist = pd.read_csv(TRAJ_DIR / "trajectory_distance.csv")
df_opt = pd.read_csv(TRAJ_DIR / "trajectory_optimal.csv")
df_rssi = pd.read_csv(TRAJ_DIR / "trajectory_rssi.csv")

datasets = {
    "distance" :df_dist,
    "optimal": df_opt,
    "rssi:": df_rssi
}

# %%
def compute_stats(df):    
    res = Result()
    num_packets = df.shape[0]

    df_conn = df[df["acked"] == True]
    df_disc = df[df["acked"] == False]

    res.plr = df_disc.shape[0] / num_packets
    res.lat_mean = df_conn["latency"].mean()
    res.lat_99 = np.percentile(df_conn["latency"].values, 99)
    res.lat_99_9 = np.percentile(df_conn["latency"].values, 99.9)
    res.num_tries_mean = df_conn["num_tries"].mean()
    res.num_tries_99 = np.percentile(df_conn["num_tries"].values, 99)
    res.num_tries_99_9 =  np.percentile(df_conn["num_tries"].values, 99)
    res.rssi_mean = df_conn["rssi"].mean()
    res.rssi_99 = np.percentile(df_conn["rssi"].values, 99)
    res.rssi_99_9 =  np.percentile(df_conn["rssi"].values, 99)
    res.pkt_roaming = df_disc[df_disc["state"] == "ROAMING"].shape[0]
    res.pkt_disconnected = df_disc[df_disc["state"] == "DISCONNECTED"].shape[0]

    return res

# %%
cols = ["name"] + [f.name for f in fields(Result)]
df_res = pd.DataFrame(columns=cols)

for name, df in datasets.items():
    res_dict = asdict(compute_stats(df))
    res_dict["name"] = name
    df_res.loc[len(df_res)] = res_dict

print(df_res)
df_res.to_csv(BASE_DIR / "result_{}.csv".format(EXP), index=False, float_format='%.4f')

# %%


# PLR
# Retransmission number
# 
# %%
