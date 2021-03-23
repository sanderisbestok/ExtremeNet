
Na Conda  install voor de originele!

pip install Cython
pip install torchvision
pip install --upgrade pip
pip install opencv-contrib-python
pip install pycocotools

cd ./external 
make

cd ..
git clone https://github.com/scaelles/DEXTR-PyTorch.git
rm -rf dextr
mv DEXTR-PyTorch dextr

Probeer dan demo