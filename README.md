# composers-assistant-REAPER
Welcome to the repository for Composer's Assistant for REAPER. 

Please watch this video to learn how Composer's Assistant works: https://www.youtube.com/watch?v=S9KdNztChx0

**INSTALLATION INSTRUCTIONS**

***STEP 1***: Install REAPER (64-bit): https://www.reaper.fm/

***STEP 2***: Install a 64-bit version of python 3.6 or higher: https://www.python.org/downloads/

***STEP 3***: Configure REAPER to see your python installation. (In REAPER, this is in the Options > Preferences > Plug-Ins > ReaScript menu)

WINDOWS USERS: If all you want to do is use the model (e.g., you do not want to train it on your own files), skip to Step 5.

***STEP 4***: Install the following packages for your python installation: pytorch and transformers

-Pytorch installation: Go to https://pytorch.org/ to get the command you need to run, then run this command from the command line. You'll want the stable build for your OS, using the "Pip" package for Python (NOT Conda). If you have an NVIDIA card, it is recommended to install a CUDA version. Otherwise, install the CPU version, which is a bit slower. The command you use should start with something like "pip3 install torch..."

-Transformers installation: From the command line, run pip install transformers

***STEP 5***: Unzip one of the following releases to your REAPER scripts folder. (To find your REAPER scripts folder, from within REAPER go to Options > Show REAPER resource path in explorer/finder..., and then open the folder called Scripts.)

Users who did step 4: Use https://github.com/m-malandro/composers-assistant-REAPER/releases/download/v1.1.2/composers.assistant.v1.1.2.zip

Windows users who skipped step 4: Use https://github.com/m-malandro/composers-assistant-REAPER/releases/download/v1.1.2/composers.assistant.v1.1.2.windows_exe.zip

Your files are in the right place if you have files like

YOUR_PATH_TO_REAPER\Scripts\composer assistant\CA_fill_empty_midi_items_in_time_selection.py

at this point.

***STEP 6***: Load the scripts into REAPER in the usual way: In REAPER, go to Actions > Show action list..., then click on New action > Load ReaScript..., and then open all scripts that start with "CA_". All other files are just helper files that these scripts need to run. Before you run any of the scripts, start the neural net server by running the composer_assistant_nn_server.py file (or, for Windows users who skipped step 4, the composer_assistant_nn_server\composer_assistant_nn_server.exe file).

**NOTE**

The models in the release were trained only on public domain and permissively-licensed MIDI files. Please see the acknowledgements, disclaimer, and license in the download.

**HOW TO CITE COMPOSER'S ASSISTANT**

Coming Soon.
