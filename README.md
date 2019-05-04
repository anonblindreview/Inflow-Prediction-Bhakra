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
Install anaconda from (http://anaconda.org) on your pc and run following commands
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
download graphviz2.38 from https://graphviz.gitlab.io/  then add its executable to $PATH variable
```python
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
```
or run following if above doesn't seem to work
```bash
pip install pydot
conda install graphviz
```
Then run LSTM/predict.py
```bash
python LSTM/predict.py
```
# Enjoy!
