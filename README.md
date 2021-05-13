:warning: **This is not ther original Extremenet**: This version is edited so it can be used for my Thesis. For the official ExtremNet repo please go to [this repository](https://github.com/xingyizhou/ExtremeNet).

## Description
This readme will provide, installation, execution and testing instructions. Experiments for the thesis are conducted on the Lisa cluster and all installations are done in an Anaconda Environment.

Algorithms used for the thesis are:

 1. [ExtremeNet](https://github.com/sanderisbestok/ExtremeNet)
 2. [TridentNet in Detectron2](https://github.com/sanderisbestok/detectron2)
 3. [YoloV5](https://github.com/sanderisbestok/yolov5)

Evaluation tools can be found in [this repository](https://github.com/sanderisbestok/thesis_tools). Data preparation steps can also be found in this repository, it is advised to first follow the steps there.


## Installation
To install on the Lisa cluster:

1. Load modules
    ```
    module load 2020
    module load Anaconda3/2020.02 
    ```

2. Clone the repo:
   ```
   git clone https://github.com/sanderisbestok/ExtremeNet
   ```

3. Create environment and install packages
   ```
   conda create --name extremenet_sander --file conda_packagelist.txt
   source activate extremenet_sander
   ```

4. Clone Dextr in the Dextr folder
   ```
   git clone https://github.com/scaelles/DEXTR-PyTorch/tree/67ca085f9509eeb2b168b07294d72f7625509fa5
   ```

5. Compile NMS
   ```
   cd ./external
   make
   ```

If installation is done on another system the same steps as described above can be used. However the prerequisites need to be installed beforehand instead of loading certain modules.

## Training
To train we need the pre-trained ExtremeNet model which can be downloaded [here](https://drive.google.com/file/d/1re-A74WRvuhE528X6sWsg1eEbMG8dmE4/view?usp=sharing). Place this model in the cache folder.

The following job can be used to train the network if the network is installed in ~/networks/extremenet_sander with the environment extremenet_sander.

```
#!/bin/bash
#SBATCH -t 04:00:00

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=gtx1080ti:4

module load 2020
module load Anaconda3/2020.02 

mkdir $TMPDIR/sander
cp -r $HOME/data $TMPDIR/sander/

source activate /home/hansen/anaconda3/envs/extremenet_sander
cd ~/networks/extremenet_sander/
python train.py ExtremeNet
```

Every 50 epochs the network will be saved, which can be used in the testing step.

## Testing
The testing can be done using the following job.

```
#!/bin/bash
#SBATCH -t 08:00:00

#SBATCH -p gpu_shared
#SBATCH -N 1
#SBATCH --gpus=1

module load 2020
module load Anaconda3/2020.02 

mkdir $TMPDIR/sander
cp -r $HOME/data $TMPDIR/sander/

source activate /home/hansen/anaconda3/envs/extremenet_sander
cd ~/networks/extremenet_sander/
python test.py ExtremeNet
```

Right now it is still hardcoded which pkl files to use. The for loop in test.py needs to be adjusted to edit which files to use.

## Extra 
ExtremeNet visualiser can be used with the following command:

python demo.py --cfg_file ExtremeNet --model_path path_to_model --demo path_to_image --show_mask


