import simpy
import random
import numpy as np
from roaming.utils import TupleRC, NetworkConfig, WifiParams
from roaming.plotter import MapPlotter
from roaming.roaming import DistanceRoaming, RSSIRoamingAlgorithm
from roaming.environment import SimpleWifiEnv, MapWifiEnv, WifiMetric, WifiStat
from roaming.trajectory import TrajectorySimulator, SimConfig
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


def main():
    # Simulation seed for reproducibility
    random.seed(SIMULATION_SEED)
    np.random.seed(SIMULATION_SEED)

    # Create simpy environment
    env = simpy.Environment()

    # Enable logging
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting simulation")


    wifi_env = MapWifiEnv(net_conf=NET_CONFIG, data_dir=DATA_FOLDER, cache_dir=CACHE_FOLDER, seed=SIMULATION_SEED)
    wifi_env.load_datasets(datasets=[("handover", "map_0")]*2 + [("handover", "map_1")]*3)

    # map_plt = MapPlotter(data_dir=DATA_FOLDER, exp_name=EXP_NAME)
    # if not map_plt.load_from_file():
    #     # wifi_sim = WifiSimulator(net_conf=NET_CONFIG, wifi_params=WIFI_PARAMS)
    #     wifi_sim = MapLoader(net_conf=NET_CONFIG)
    #     wifi_sim.load_maps(
    #         wifi_metric=WifiMetric.LATENCY,
    #         wifi_stats=WifiStat.MEAN,
    #         map_files=[FMAP16, FMAP16, FMAP17, FMAP17, FMAP17]*5
    #     )
    #     map_plt.set_simulator(wifi_sim)
    # if not map_plt.map_loaded:
    #     map_plt.generate_maps(num_samples=10, save=False)
    # map_plt.plot_maps()

    # wifi_sim = WifiSimulator(net_conf=NET_CONFIG, wifi_params=WIFI_PARAMS)
    # #roam_alg = DistanceRoaming(env=env, wifi_sim=wifi_sim, roaming_time=0.2)
    # roam_alg = RSSIRoamingAlgorithm(env=env, wifi_sim=wifi_sim, roaming_time=0.2, rssi_threshold=-85)

    # sim_config = SimConfig(exp_name=EXP_NAME, pkt_period=0.1, speed=0.5)
    # traj_sim = TrajectorySimulator(env=env, wifi_sim=wifi_sim, roam_alg=roam_alg)
    # traj_sim.generate_trajectory(50)
    # traj_sim.configure(sim_config)
    # env.run()

    # print(wifi_sim.ap_positions)

    # map_plt = MapPlotter(data_dir=DATA_FOLDER, exp_name=EXP_NAME)
    # map_plt.load_from_file()
    # map_plt.plot_maps()



if __name__ == "__main__":
    main()


# TODO list
# fare codice che croppa matrici quando sono sotto rssi minimo

# create wifi environment that loads ns-3 maps (aggregated)
# understand how to compute SNR/RSSI
# simulate map in ns-3

# add metric computation from simulated trajectory
# implement optimal switching technique

# fix RSSI simulator (to avoid switching when also other APs are bad)

# add type hints to functions
# fix logging messages
