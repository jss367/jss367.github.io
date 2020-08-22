
Linux/Mac

git clone https://github.com/pdollar/coco.git

cd coco/PythonAPI
make
sudo make install
sudo python setup.py install





Windows
This is what worked for me:

conda install -c anaconda cython
pip install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI