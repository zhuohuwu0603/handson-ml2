# T81 558:Applications of Deep Neural Networks
[Washington University in St. Louis](http://www.wustl.edu)

Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/)

**The content of this course changes as technology evolves**, to keep up to date with changes [follow me on GitHub](https://github.com/jeffheaton).

* Section 2. Fall 2019, Monday, 2:30 PM - 5:20 PM Online & Duncker / 101 
* Section 1. Fall 2019, Monday, 6:00 PM - 9:00 PM Online & Cupples II / L009

# Course Description




conda create -y --name tf2_py36 python=3.6

source activate tf2_py36

conda install -y jupyter

conda install -y scipy
pip install --exists-action i --upgrade sklearn
pip install --exists-action i --upgrade pandas
pip install --exists-action i --upgrade pandas-datareader
pip install --exists-action i --upgrade matplotlib
pip install --exists-action i --upgrade pillow
pip install --exists-action i --upgrade tqdm
pip install --exists-action i --upgrade requests
pip install --exists-action i --upgrade h5py
pip install --exists-action i --upgrade pyyaml
pip install --exists-action i --upgrade tensorflow_hub
pip install --exists-action i --upgrade bayesian-optimization
pip install --exists-action i --upgrade spacy
pip install --exists-action i --upgrade gensim
pip install --exists-action i --upgrade flask
pip install --exists-action i --upgrade boto3
pip install --exists-action i --upgrade gym
pip install --exists-action i --upgrade tensorflow==2.0.0-beta1
pip install --exists-action i --upgrade keras-rl2 --user


jupyter notebook

conda update -y --all

python -m ipykernel install --user --name tf2_py36 --display-name "tf2_py36 (tensorflow2b1_python36)"


ipython kernelspec uninstall python3



2. convert jupter notebook to python file
    ls *ipynb | jupyter nbconvert *ipynb --to python
    
3. Install new 

pip install -U seaborn

pip install opencv-python

pip install statsmodels

pip install httplib2

pip install BeautifulSoup4
pip install lxml
pip install html5lib 
 
pip install pyspark==2.4.4 

pip install flask

pip install xgboost

# xgboost for classification
pip install lime missingno pydotplus shap yellowbrick yellowbrick xgbfir pdpbox

pip install keras

pip install 2to3
pip install progressbar
pip install terminaltables
pip install cvxopt

pip install lime missingno pydotplus shap yellowbrick yellowbrick xgbfir pdpbox
pip install pydotplus shap yellowbrick xgbfir pdpbox
