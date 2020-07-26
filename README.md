### Higlights
* Each site can only access one user specified GPU at the moment. 
* GPU for each site can be specified in inputspec.json as gpus:[0, 1, 2...]. Empty list/or no gpu options means use CPU.
* For majority of classification tasks, one should only change:
    * models.py, if they wish to change the architecture
    * local.py to parse the desired dataset.
    * Multi class classification uses Confusion matrix and micro/macro F1 scores(see core/measurements.py). 
* For regression tasks, one should change the above plus:
    * core/nn.py, iteration(...)
    * local.py, init_cache(...)
* pooled.py is an easy way of running a pooled experiment on all site without out having to move data in any sort. It will internally pool data from all site and feed to the network.

![DINUNET](assets/dinunet.png)
### Structure
* [dinunet/assets](https://github.com/trendscenter/dinunet/tree/master/assets)
* [dinunet/<input_folder_placeholder>](https://github.com/trendscenter/dinunet/tree/master/test) An example for input folder format.
    * [dinunet/inputspec.json](https://github.com/trendscenter/dinunet/blob/master/test/inputspec.json) Input specification for each sites as list of json.
* [dinunet/core](https://github.com/trendscenter/dinunet/tree/master/core)
    * [dinunet/measurements.py](https://github.com/trendscenter/dinunet/blob/master/core/measurements.py) GPU implementation of various metrics like Precision, Recall, F<sub>beta</sub>, Accuracy, IOU, and Confusion Matrix
    * [dinunet/nn.py](https://github.com/trendscenter/dinunet/blob/master/core/nn.py) All neural network related utilities goes here.
    * [dinunet/utils.py](https://github.com/trendscenter/dinunet/blob/master/core/utils.py) General utilities.
* [dinunet/compspec.json](https://github.com/trendscenter/dinunet/blob/master/compspec.json) Computation specification for COINSTAC with gpu options.
* [dinunet/Dockerfile](https://github.com/trendscenter/dinunet/blob/master/Dockerfile) Dockerfile capable of GPU access.
* [dinunet/local.py](https://github.com/trendscenter/dinunet/blob/master/local.py) All computation to be run in every sites.
* [dinunet/remote.py](https://github.com/trendscenter/dinunet/blob/remote.py) Receives some information from each sites and sends feed back iteration by iteration.
* [dinunet/pooled.py](https://github.com/trendscenter/dinunet/blob/pooled.py) Easy way to run experiments by pooling data of all sites (Only developers might use this feature).
* [dinunet/requirements.txt](https://github.com/trendscenter/dinunet/blob/master/requirements.txt) Python dependencies.

# Usage
* It needs nodejs, so I recommend using [NVM](https://github.com/nvm-sh/nvm) for hassle free installation.
* [coinstac-simulator](https://github.com/trendscenter/coinstac/tree/master/packages/coinstac-simulator)
* [coinstac](https://github.com/trendscenter/coinstac)

