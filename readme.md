# Qule
Qule is a quantum enhanced AI driven drug design tool. 

This library refers to the following source code.
* [yongqyu/MolGAN-pytorch](https://github.com/yongqyu/MolGAN-pytorch)


## Dependencies

* **python>=3.7**
* **pytorch>=0.4.1**: https://pytorch.org
* **rdkit**: https://www.rdkit.org
* **pennylane**
* **tensorflow==1.15**
* **frechetdist**

## Dataset
* run bash script `data/gdb9_generater.sh` to download gdb database and then run `data/sparse_molecular_dataset.py` to generate molecular graph dataset used to train the model.

## Training
```
python main.py --mode=train

```

## Prediction
To run the model against test dataset, make sure the model is fully trainned in the first place.
```
python main.py --mode=test
```
## Structure
`main.py` parse the command line arguments and pass it to the `Qgans_molGen.py` which access generator and discriminator model from `models.py` which inturn access `layers.py` and `utils.py` evaluate the metrics.  

Below are some generated molecules:
[![generated-sample.png](https://i.postimg.cc/RVKvm6B2/generated-sample.png)](https://postimg.cc/F7rMgK1x)


## Webapp
Here are some snapshots of the WebApplication:
[![39.png](https://i.postimg.cc/j2cMb4MR/39.png)](https://postimg.cc/p59fQzNS)
[![40.png](https://i.postimg.cc/tg6DTjBS/40.png)](https://postimg.cc/Z9T671tp)
[![41.png](https://i.postimg.cc/1tcHGKhD/41.png)](https://postimg.cc/jw2PKPjj)
[![42.png](https://i.postimg.cc/NFsxprBS/42.png)](https://postimg.cc/9RkTf04Y)



