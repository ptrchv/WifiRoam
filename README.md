# WifiRoam

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

## Data
- [Zenono dataset](https://doi.org/10.5281/zenodo.17938018) (to download inside the ```data``` folder).
