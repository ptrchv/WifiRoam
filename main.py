# %%
import simpy
import random
import numpy as np
from roaming.utils import TupleRC, NetworkConfig, WifiParams
from roaming.plotter import MapPlotter
from roaming.roaming import DistanceRoaming, RSSIRoamingAlgorithm, OptimizedRoaming
from roaming.environment import SimpleWifiEnv, MapWifiEnv
from roaming.trajectory import TrajectorySimulator, SimConfig
from roaming.metrics import WifiMetric, WifiStat
import logging

logger = logging.getLogger(__name__)


DATA_FOLDER = "data"
CACHE_FOLDER = "cache"

EXP_NAME = "test_ns3_maps"
SIMULATION_SEED = 2

MAP_DIMS = TupleRC(60, 120)
NET_CONFIG = NetworkConfig(
    map_dims = MAP_DIMS,
    ap_positions = [
        TupleRC(0, 0),                              # Top-Left
        TupleRC(MAP_DIMS.row-1, 0),                 # Bottom-Left
        TupleRC(0, MAP_DIMS.col-1),                 # Top-Right
        TupleRC(MAP_DIMS.row-1, MAP_DIMS.col-1),    # Bottom-Right
        TupleRC(MAP_DIMS.row//2, MAP_DIMS.col//2)   # Center
    ],
    ap_loads = [0.2, 0.2, 0.2, 0.2, 0.6]
)

WIFI_PARAMS = WifiParams (
    rssi_threshold = -85.0,
    handover_penalty = 500,
    switch_penalty = 50,
    no_ap_penalty = 2000,
)


# %%
def main():
    # Simulation seed for reproducibility
    random.seed(SIMULATION_SEED)
    np.random.seed(SIMULATION_SEED)

    # Create simpy environment
    env = simpy.Environment()

    # Enable logging
    logging.basicConfig(level=logging.INFO)

    # Create wifi environment
    #wifi_env = SimpleWifiEnv(net_conf=NET_CONFIG, wifi_params=WIFI_PARAMS)
    wifi_env = MapWifiEnv(net_conf=NET_CONFIG, data_dir=DATA_FOLDER, cache_dir=CACHE_FOLDER, seed=SIMULATION_SEED)
    wifi_env.load_datasets(datasets=[("handover_hi_res", "map_0")]*2 + [("handover_hi_res", "map_1")]*3)

    # Create roaming algorithm
    # roam_alg = DistanceRoaming(env=env, wifi_sim=wifi_env, roaming_time=0.2)
    roam_alg = RSSIRoamingAlgorithm(env=env, wifi_sim=wifi_env, roaming_time=0.2, rssi_threshold=-80)
    #roam_alg = OptimizedRoaming(env, wifi_sim=wifi_env, roaming_time=0.2, metric=WifiMetric.NUM_TRIES, stat=WifiStat.MEAN)

    # Create trajectory simulator
    sim_config = SimConfig(pkt_period=0.1, speed=0.5, beacon_time=0.1)
    traj_sim = TrajectorySimulator(env=env, wifi_sim=wifi_env, roam_alg=roam_alg, sim_config=sim_config, cache_dir=CACHE_FOLDER, exp_name=EXP_NAME)

    # Set trajectory sim
    #roam_alg.configure(traj_sim)

    # Configure trajectory simulator
    traj_sim.generate_trajectory(50)
    traj_sim.configure()

    # Run simulation
    logging.info("Starting simulation")
    env.run()
    logging.info("Simulation Ended")

    # Trajectory plots
    map_plt = MapPlotter(wifi_sim=wifi_env, cache_dir=CACHE_FOLDER, exp_name=EXP_NAME)
    map_plt.generate_maps()
    map_plt.plot_maps(trajectory=True)


if __name__ == "__main__":
    main()


# TODO list
# fix structure of cache folder (common/experiments)
# fix RSSI simulator (to avoid switching when also other APs are bad)
# add type hints to functions
# fix logging messages
# fix configuration
# sfasare beacons rispetto a messaggi

# implement packet queuing when when you roam or disconnect (for more realistic latency) -> useful also if optimizing another metric
# latency is a problem since if is hard to re-compute averages and percentiles -> you could limit the number of switches


# OPTIMAL ROAMING improvements
# for every switch point decide whether to switch
    # - (you can put limit on switches)
# you need to take into account
# - lost packets due to switch
# - remove switch point that may be too close if you select one (time to reach lower than roaming time)
# search is like exploring a tree (where you have a counter on the number of switches)
# if too many switch set a threshold for minimum metrics different that may consider the need of switching
# add events for roaming on the path (roaming period can be placed in between, packets, to avoid having lost packets)
# queue dropped packets during roaming

