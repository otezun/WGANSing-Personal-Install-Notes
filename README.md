# Personal Notes concerning WGANSing

Word of Warning: I have no idea what I'm doing here. I am also in no way affiliated with WGANSing. 


A tutorial on how to set up and use MTG/WGANSing (https://github.com/MTG/WGANSing) on Windows 10.
This tutorial assumes you have downloaded WGANSing already did the following:
"To use the WGANSing, you will have to download the model weights and place it in the log_dir directory, defined in config.py.

The NUS-48E dataset can be downloaded from here. Once downloaded, please change wav_dir_nus in config.py to the same directory that the dataset is in." (from WGANSing README).

# Hardware Requirements
I recommend a good GPU with CUDA support, personally I use a MSI GTX 1660 Armor OC with a Ryzen 5 3600 and 64 GB of RAM - slightly overclocked.
OBVIOUSLY, you will need the NVIDIA drivers to run TensorFlow with CUDA.

# Basic Setup
This is Windows only at the moment. You will need Anaconda (https://www.anaconda.com/). 
While Anaconda is not strictly necessary it does help make things easier in the long run.
I tried to set it up on linux, but ran into issues with the NVIDIA drivers and CUDA. 

1. Open Anaconda Prompt
2. "conda create -n myenv python=3.6" to create conda environment with python 3.6 which is necessary for TensorFlow 1.15.x (legacy)
3. "conda install tensorflow-gpu=1.15" to install tensorflow 1.15
4. Open WGANSing-mtg/requirements.txt in an editor such as Notepad++ and remove version requirements
for h5py, librosa, mir-eval, numpy, pandas and cython
5. "pip install -r requirements.txt"
It should now install the requirements without breaking. This worked for my machine. 
7. Create directories for wav_dir_nus, voice_dir and log_dir according to preference
8. In WGANSing-mtg/config.py set the wav_dir_nus, voice_dir and log_dir
9. Now when using "python main.py -t" it will still send error messages, since matplotlib, tqdm, pysptk are not installed yet, so install them using pip (not ideal when working with conda, but it works)
10. Open "data_pipeline.py" and comment out the assert function on line 130 (also not ideal, but works):         
#assert feats_targs.max()<=1.0 and feats_targs.min()>=0.0
11. Run the prep_data_nus.py, this will take some time
12. Run "python main.py -t"
It should now train the model data using 500-1000 epochs. There may be crashes. In that case just run it again. It seems to save at least some training data before a crash, allowing it to restore +-100 epochs before the crash.

This is how far I have got so far.

# Other comments
Currently it seems that WGANSing is in the process of updating to TensorFlow 2. The most important issues are listed here: https://github.com/MTG/WGANSing/issues/19

