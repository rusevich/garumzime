# garumzime 
garumzime - transformer-based orthography checker for Latvian

## PLAN:
* data.py - download data, prepare data, serve data
* model.py + config.py - expandable transformer structure for a model
* train.py - training functions
* eval.py - evaluation functions


After this, scripts:
* run_tranining.py - read config, feed to train.py, evaluate with eval.py 
* run_inference.py - use trained model


### First step
data.py and data_sources need to be done