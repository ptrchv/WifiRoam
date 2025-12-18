# Information
Two network configurations, representative of an industrial environment, are simulated using the ns-3 library.
A network configuration consists of an empty 2D space with an access point (AP) positioned in the middle.
A Wi-Fi station (STA), designated as the Station Under Test (SUT), is statically positioned in different locations of the 2D space according to a grid.
Additional STAs connected to the AP are positioned in the simulated space to act as interferers.
For each location of the SUT, an independent simulation is carried out to build a 2D map of the environment.
The grid spacing defines the resolution of the map.

Two campaigns were conducted for each network configuration using different resolutions: 1 m x 1 m (*handover_hi_res*) and 2.5 m x 2.5 m (*handover*).

More information about the simulation parameters can be found in a previously published article in Section IV-B *Simulation setup*: [https://doi.org/10.1109/WFCS63373.2025.11077620](https://doi.org/10.1109/WFCS63373.2025.11077620).

Each campaign folder (*handover* and *handover_hi_res*) contains the raw data (*raw*), the extracted datasets (*datasets*), and two .json files describing the network configurations. *config.json* contains the simulation parameters and associates each dataset file (the output of a single simulation) with the position of the SUT in the 2D environment.

Inside the *datasets* folders, the two network configurations are separated (map_0, map_1), and only the most relevant data is extracted.
For each packet sent by the SUT, the following information is stored:
* Sequence number (`seq`)
* Whether the frame delivery was successful (`acked`)
* Latency: time from packet generation to the reception of the acknowledgement or dropping of the packet (`latency`)
* Number of frame transmissions required (`transmissions`)
* RSSI and noise level of the last beacon received before the start of transmission (`rssi`, `noise`)
Empty files usually denote that the simulation was non possibile due to the SUT being out of range of the AP.

Starting from the raw data, everything else can be extracted using the `extractor.py` script.