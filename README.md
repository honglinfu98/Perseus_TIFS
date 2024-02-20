# Perseus
Inspired by Perseus’s clever and strategic approach to challenges, this repository meticulously analyzes information to expose the mastermind behind crypto market manipulation.


### Set up the repo

#### Create a python virtual environment

- macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
pip3 install -r dev-requirements.txt
pip3 install -e .
export $(cat .env | xargs)
```

- Windows

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r dev-requirements.txt
pip install -e .
export $(cat .env | xargs)
```

# Run the torch install

Run the Jupyter notebook torch_install.ipynb to install torch dependencies

### For running 

```bash
python3 main.py
```

