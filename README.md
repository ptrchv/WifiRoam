# WifiRoaming

## plt() show with Remote-SSH (VSCode)
Check on server if X11Forwarding is enabled (`X11Forwarding yes`):

```bash
vim /etc/ssh/sshd_config
# uncomment "X11Forwarding yes"
```

Enable X11Forwarding on client connection (add `ForwardX11 yes` in host configuration):
```bash
vim ~/.ssh/config
# add `ForwardX11 yes` in host configuration
```

Install PyQt5 on server to fix warning (possibly in virtual environment):
```bash
python -m pip install PyQt5
```




# Material
- [DAWN in Open WRT](https://openwrt.org/docs/guide-user/network/wifi/dawn)
- [Shared folder Onedrive](https://cnrsc-my.sharepoint.com/:f:/g/personal/pietrochiavassa_cnr_it/EpMWGTvyLUZIg7SypWAb1lIBpv4MWQ1yAclc3aNCDxHS6A?e=2wSiFC)



 # Open questions
 - how latency is computed?
 - how handover penalty is taken into account when training


 ## Todo

 - generate map plots from boris code (possibily one plot for each metric, RSSI and latency, with all APs)
 - function to generate trajectories
 - function to generate handover times