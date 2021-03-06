# Inflow-Prediction-Bhakra
Inflow determination of Bhakra Reservoir using Long Short-Term Memory network
# System Requirements 
```bash
python 3.6+
tensorflow-gpu
keras
pandas
numpy
matplotlib
sklearn
```
# How to run
Install anaconda from [this](http://anaconda.org) on your pc and run following commands
```bash
conda create --name tf-gpu
conda activate tf-gpu
conda install -c aaronzs tensorflow-gpu
conda install -c anaconda cudatoolkit
conda install -c anaconda cudnn
conda install keras-gpu
pip install pandas
pip install sklearn
pip install matplotlib
```
Then run to download the git repository
```bash
git clone https://github.com/Anurag14/Inflow-Prediction-Bhakra
cd Inflow-Prediction-Bhakra
```
# To auto generate model graphs
download graphviz2.38 from [here](https://graphviz.gitlab.io/)  then add its executable to $PATH variable.
```python
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
```
or run following if above doesn't seem to work
```bash
pip install pydot_ng
conda install graphviz
```
#### TL;DR install graphviz via conda and add its path to $PATH instead of building it from base.
#### Update: As of writing this library there seems to be a open issue [here](https://github.com/keras-team/keras/issues/12640)
Same quick fix is applied by hacking into the keras library ../keras/utils/vis_utils.py and change 
```python
 import pydot
``` 
to 
```python
import pydot_ng as pydot
```
Then run LSTM/predict.py
```bash
python LSTM/predict.py
```
# Enjoy!
