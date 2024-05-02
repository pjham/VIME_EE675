# Setup instructions

Use python3.9, to install - `brew install python@3.9`
## Virtual environment

Navigate to the code directory in the terminal and run the following commmands:
```
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Execute

To train models with VIME, keep `--vime 1` else make it `--vime 0`:

```
  python run.py --env Pendulum-v0 --buffer 100000 --n_epi 250 --vime 1
```
OR
```
python run.py --env MountainCarContinuous-v0 --buffer 100000 --n_epi 250 --vime 1
```
