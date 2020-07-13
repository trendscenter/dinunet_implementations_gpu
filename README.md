![DINUNET](assets/dinunet.png)

### Structure
* [dinunet_vbm/assets](https://github.com/trendscenter/dinunet_vbm/tree/master/assets)
* [dinunet_vbm/<input_folder_placeholder>](https://github.com/trendscenter/dinunet_vbm/tree/master/test) An example for input folder format.
    * [dinunet_vbm/inputspec.json](https://github.com/trendscenter/dinunet_vbm/blob/master/test/inputspec.json) Input specification for each sites as list of json.
* [dinunet_vbm/core](https://github.com/trendscenter/dinunet_vbm/tree/master/core)
    * [dinunet_vbm/datautils.py](https://github.com/trendscenter/dinunet_vbm/blob/master/core/data_parser.py) is where the data related logic goes.
    * [dinunet_vbm/measurements.py](https://github.com/trendscenter/dinunet_vbm/blob/master/core/measurements.py) GPU implementation of various metrics like Precision, Recall, F<sub>beta</sub>, Accuracy, IOU, and Confusion Matrix
    * [dinunet_vbm/models.py](https://github.com/trendscenter/dinunet_vbm/blob/master/core/models.py) All neural network models used are to be listed here.
    * [dinunet_vbm/utils.py](https://github.com/trendscenter/dinunet_vbm/blob/master/core/utils.py) General utilities
* [dinunet_vbm/classification.py](https://github.com/trendscenter/dinunet_vbm/blob/master/classification.py) All logic involving loss function, training, evaluation goes here.
* [dinunet_vbm/compspec.json](https://github.com/trendscenter/dinunet_vbm/blob/master/compspec.json) Computation specification for COINSTAC.
* [dinunet_vbm/Dockerfile](https://github.com/trendscenter/dinunet_vbm/blob/master/Dockerfile) Dockerfile capable of GPU access.
* [dinunet_vbm/local.py](https://github.com/trendscenter/dinunet_vbm/blob/master/local.py) All computation to be run in every sites.
* [dinunet_vbm/remote.py](https://github.com/trendscenter/dinunet_vbm/blob/remote.py) Receives some information from each sites and sends feed back iteration by iteration.
* [dinunet_vbm/requirements.txt](https://github.com/trendscenter/dinunet_vbm/blob/master/requirements.txt) Python dependencies.