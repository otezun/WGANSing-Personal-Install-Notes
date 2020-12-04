# Personal Notes concerning WGANSing

Word of Warning: I have no idea what I'm doing here. I am also in no way affiliated with WGANSing. 


A tutorial on how to set up and use MTG/WGANSing (https://github.com/MTG/WGANSing) on Windows 10.
This tutorial assumes you have downloaded WGANSing already did the following:
"To use the WGANSing, you will have to download the model weights and place it in the log_dir directory, defined in config.py.

The NUS-48E dataset can be downloaded from here. Once downloaded, please change wav_dir_nus in config.py to the same directory that the dataset is in." (from WGANSing README).

# Hardware Requirements
I recommend a good GPU with CUDA support, personally I use a MSI GTX 1660 Armor OC with a Ryzen 5 3600 and 64 GB of RAM - slightly overclocked.
OBVIOUSLY, you will need the NVIDIA drivers to run TensorFlow with CUDA. It took more than 12 hours to train it on the nus dataset with this setup.

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
13. create the val_dir_synth directory as specified in the config.py

Update: It now has trained the model, taking more than 12 hours (I don't know the exact time as I went to sleep but at least it didn't crash and was finished when I woke up). The official documentation states that it requires a .lab file as an input, however, testing it with a .lab file did not work:
`Currently only supporting hdf5 files which are in the dataset, will be expanded later.
['nus_ADIZ_read_01.hdf5', 'nus_ADIZ_read_09.hdf5', 'nus_ADIZ_read_13.hdf5', 'nus_ADIZ_read_18.hdf5', 'nus_ADIZ_sing_01.hdf5', 'nus_ADIZ_sing_09.hdf5', 'nus_ADIZ_sing_13.hdf5', 'nus_ADIZ_sing_18.hdf5', 'nus_JLEE_read_05.hdf5', 'nus_JLEE_read_08.hdf5', 'nus_JLEE_read_11.hdf5', 'nus_JLEE_read_15.hdf5', 'nus_JLEE_sing_05.hdf5', 'nus_JLEE_sing_08.hdf5', 'nus_JLEE_sing_11.hdf5', 'nus_JLEE_sing_15.hdf5', 'nus_JTAN_read_07.hdf5', 'nus_JTAN_read_15.hdf5', 'nus_JTAN_read_16.hdf5', 'nus_JTAN_read_20.hdf5', 'nus_JTAN_sing_07.hdf5', 'nus_JTAN_sing_15.hdf5', 'nus_JTAN_sing_16.hdf5', 'nus_JTAN_sing_20.hdf5', 'nus_KENN_read_04.hdf5', 'nus_KENN_read_10.hdf5', 'nus_KENN_read_12.hdf5', 'nus_KENN_read_17.hdf5', 'nus_KENN_sing_04.hdf5', 'nus_KENN_sing_10.hdf5', 'nus_KENN_sing_12.hdf5', 'nus_KENN_sing_17.hdf5', 'nus_MCUR_read_04.hdf5', 'nus_MCUR_read_10.hdf5', 'nus_MCUR_read_12.hdf5', 'nus_MCUR_read_17.hdf5', 'nus_MCUR_sing_04.hdf5', 'nus_MCUR_sing_10.hdf5', 'nus_MCUR_sing_12.hdf5', 'nus_MCUR_sing_17.hdf5', 'nus_MPOL_read_05.hdf5', 'nus_MPOL_read_11.hdf5', 'nus_MPOL_read_19.hdf5', 'nus_MPOL_read_20.hdf5', 'nus_MPOL_sing_05.hdf5', 'nus_MPOL_sing_11.hdf5', 'nus_MPOL_sing_19.hdf5', 'nus_MPOL_sing_20.hdf5', 'nus_MPUR_read_02.hdf5', 'nus_MPUR_read_03.hdf5', 'nus_MPUR_read_06.hdf5', 'nus_MPUR_read_14.hdf5', 'nus_MPUR_sing_02.hdf5', 'nus_MPUR_sing_03.hdf5', 'nus_MPUR_sing_06.hdf5', 'nus_MPUR_sing_14.hdf5', 'nus_NJAT_read_07.hdf5', 'nus_NJAT_read_15.hdf5', 'nus_NJAT_read_16.hdf5', 'nus_NJAT_read_20.hdf5', 'nus_NJAT_sing_07.hdf5', 'nus_NJAT_sing_15.hdf5', 'nus_NJAT_sing_16.hdf5', 'nus_NJAT_sing_20.hdf5', 'nus_PMAR_read_05.hdf5', 'nus_PMAR_read_08.hdf5', 'nus_PMAR_read_11.hdf5', 'nus_PMAR_read_15.hdf5', 'nus_PMAR_sing_05.hdf5', 'nus_PMAR_sing_08.hdf5', 'nus_PMAR_sing_11.hdf5', 'nus_PMAR_sing_15.hdf5', 'nus_SAMF_read_01.hdf5', 'nus_SAMF_read_09.hdf5', 'nus_SAMF_read_13.hdf5', 'nus_SAMF_read_18.hdf5', 'nus_SAMF_sing_01.hdf5', 'nus_SAMF_sing_09.hdf5', 'nus_SAMF_sing_13.hdf5', 'nus_SAMF_sing_18.hdf5', 'nus_VKOW_read_05.hdf5', 'nus_VKOW_read_11.hdf5', 'nus_VKOW_read_19.hdf5', 'nus_VKOW_read_20.hdf5', 'nus_VKOW_sing_05.hdf5', 'nus_VKOW_sing_11.hdf5', 'nus_VKOW_sing_19.hdf5', 'nus_VKOW_sing_20.hdf5', 'nus_ZHIY_read_02.hdf5', 'nus_ZHIY_read_03.hdf5', 'nus_ZHIY_read_06.hdf5', 'nus_ZHIY_read_14.hdf5', 'nus_ZHIY_sing_02.hdf5', 'nus_ZHIY_sing_03.hdf5', 'nus_ZHIY_sing_06.hdf5', 'nus_ZHIY_sing_14.hdf5']`
From the line "Currently only supporting hdf5 files which are in the dataset, will be expanded later" we can figure out that we need a .hdf5 instead. 
`python main.py -e nus_ZHIY_sing_06.hdf5 MPOL`
This will take the sung voice of ZHIY (male voice) and translate it into MPOL (female voice). Then it will display a beautiful graph which you can technically save, although I don't remember where I saved it to. It will then ask to synthesize the output, which we answer with a solid Y(es).
It will now write the file to disk. We can also synthesize the ground truth using vocoder, giving the thing better quality.
# Other comments
Currently it seems that WGANSing is in the process of updating to TensorFlow 2. The most important issues are listed here: https://github.com/MTG/WGANSing/issues/19
