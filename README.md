
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt -y update
sudo apt install -y python3.11
sudo apt install -y python3.11-venv

sudo apt-get install python3.11-dev
```

```
python3.11 -m venv event_env
source event_env/bin/activate
pip3 install -r requirements.txt

export PROJECT_ROOT=$(pwd)

## setting dataset
mkdir datasets
ln -s /path/to/dataset ./datasets/DATA
```
