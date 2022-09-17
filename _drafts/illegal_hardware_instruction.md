hello

In [1]: import tensorflow
[1]    82655 illegal hardware instruction  ipython
(bv2) ➜  ~ which python3
/opt/homebrew/bin/python3
(bv2) ➜  ~ which python
/Users/julius/opt/anaconda3/envs/bv2/bin/python
(bv2) ➜  ~



(bv2) ➜  ~ file $(which python3)
/opt/homebrew/bin/python3: Mach-O 64-bit executable arm64
(bv2) ➜  ~ file $(which python)
/Users/julius/opt/anaconda3/envs/bv2/bin/python: Mach-O 64-bit executable x86_64
(bv2) ➜  ~


(base) ➜  ~ conda deactivate
➜  ~ conda deactivate
➜  ~ file $(which /usr/bin/python3)
/usr/bin/python3: Mach-O universal binary with 2 architectures: [x86_64:Mach-O 64-bit executable x86_64
- Mach-O 64-bit executable x86_64] [arm64e:Mach-O 64-bit executable arm64e
- Mach-O 64-bit executable arm64e]
/usr/bin/python3 (for architecture x86_64):	Mach-O 64-bit executable x86_64
/usr/bin/python3 (for architecture arm64e):	Mach-O 64-bit executable arm64e
➜  ~

arch -arm64 bash install_venv.sh --python=/usr/bin/python3 my_tf_env

arch -arm64 bash install_venv.sh --python=/usr/bin/python3 bv2


/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/chinitaberrio/tensorflow_macos/master/scripts/download_and_install.sh)"

ERROR: TensorFlow with ML Compute acceleration is only available on macOS 11.0 and later.


/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/apple/tensorflow_macos/master/scripts/download_and_install.sh)"



If you got the wrong installer, you'll need to:
conda install anaconda-clean
anaconda-clean --yes



rm -rf ~/anaconda3 might delete it from some places, but if you do `where conda`, you might get `/Users/julius/opt/anaconda3/condabin/conda`. so
`rm -rf /Users/julius/opt/anaconda3`


If you get this: CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
To initialize your shell, run

find your base env: conda info | grep -i 'base environment'


source ~/anaconda3/etc/profile.d/conda.sh
source ~/opt/anaconda3/etc/profile.d/conda.sh



