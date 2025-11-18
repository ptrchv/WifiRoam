import simpy
import random
import numpy as np
from roaming.utils import TupleRC, NetworkConfig, WifiParams
from roaming.plotter import MapPlotter
from roaming.roaming import DistanceRoaming, RSSIRoamingAlgorithm
from roaming.environment import WifiSimulator
from roaming.trajectory import TrajectorySimulator, SimConfig
import logging

logger = logging.getLogger(__name__)


DATA_FOLDER = "data"
EXP_NAME = "test3"
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

    # map_plt = MapPlotter(data_dir=DATA_FOLDER, exp_name=EXP_NAME)
    # if not map_plt.load_from_file():
    #     wifi_sim = WifiSimulator(net_conf=NET_CONFIG, wifi_params=WIFI_PARAMS)
    #     map_plt.set_simulator(wifi_sim)
    # if not map_plt.map_loaded:
    #     map_plt.generate_maps()        
    # map_plt.plot_maps()

    wifi_sim = WifiSimulator(net_conf=NET_CONFIG, wifi_params=WIFI_PARAMS)
    #roam_alg = DistanceRoaming(env=env, wifi_sim=wifi_sim, roaming_time=0.2)
    roam_alg = RSSIRoamingAlgorithm(env=env, wifi_sim=wifi_sim, roaming_time=0.2, rssi_threshold=-85)

    sim_config = SimConfig(exp_name=EXP_NAME, pkt_period=0.1, speed=0.5)    
    traj_sim = TrajectorySimulator(env=env, wifi_sim=wifi_sim, roam_alg=roam_alg)
    traj_sim.generate_trajectory(50)
    traj_sim.configure(sim_config)
    env.run()

    print(wifi_sim.ap_positions)

    map_plt = MapPlotter(data_dir=DATA_FOLDER, exp_name=EXP_NAME)
    map_plt.load_from_file()
    map_plt.plot_maps()
    


if __name__ == "__main__":    
    main()


# TODO list
# finish trajectory simulator
# fare codice che croppa matrici quando sono sotto rssi minimo
# controllare che AP siano nel posto giusto
# disegnare delle mappe e capire le dimensioni e copertura
# fare simulatore traiettorie
# simulare in ns-3 una mappa su cui fare esperimenti reali
# add type hints to functions
# change TupleRC to float
# plot trajectories