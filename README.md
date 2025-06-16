# composers-assistant-REAPER
Welcome to the repository for Composer's Assistant for REAPER. 

Please watch this video to learn how Composer's Assistant works: https://www.youtube.com/watch?v=S9KdNztChx0

Please watch this video to learn about the new features introduced in version 2: https://www.youtube.com/watch?v=NZmvbxR3LFM

Please watch this video if you need help with the installation instructions below: https://www.youtube.com/watch?v=kbc1Awf0sM0

**INSTALLATION INSTRUCTIONS**

***STEP 1***: Install REAPER (64-bit): https://www.reaper.fm/

***STEP 2***: Install a 64-bit version of python 3.9 or higher: https://www.python.org/downloads/

***STEP 3***: Configure REAPER to see your python installation. (In REAPER, this is in the Options > Preferences > Plug-Ins > ReaScript menu)

WINDOWS USERS: If all you want to do is use the model (e.g., you do not want to train it on your own files), skip to Step 5.

***STEP 4***: Install the following packages for your python installation: pytorch and transformers

-Pytorch installation: Go to https://pytorch.org/ to get the command you need to run, then run this command from the command line. You'll want the stable build for your OS, using the "Pip" package for Python (NOT Conda). If you have an NVIDIA card, it is recommended to install a CUDA version. Otherwise, install the CPU version, which is a bit slower. The command you use should start with something like "pip3 install torch..."

-Transformers installation: From the command line, run pip install transformers

***STEP 5***: Unzip one of the following releases to your REAPER resources folder. (To find your REAPER resources folder, from within REAPER go to Options > Show REAPER resource path in explorer/finder...)

Users who did step 4: Use https://github.com/m-malandro/composers-assistant-REAPER/releases/download/v2.1.0/composers.assistant.v.2.1.0.zip

Windows users who skipped step 4: Use https://github.com/m-malandro/composers-assistant-REAPER/releases/download/v2.1.0/composers.assistant.v.2.1.0.windows.exe.zip

Your files are in the right place if you have files like

YOUR_PATH_TO_REAPER\Scripts\composers_assistant_v2\composers_assistant_nn_server.py

and

YOUR_PATH_TO_REAPER\Effects\composers_assistant_v2\CAv2 Global Options

at this point.

***STEP 6***: Restart REAPER. Load the scripts into REAPER in the usual way: In REAPER, go to Actions > Show action list..., then click on New action > Load ReaScript..., and then open all three scripts that start with "REAPER_". All other files are just helper files that these scripts need to run. Before you run any of the scripts, start the neural net server by running the composers_assistant_nn_server.py file in the scripts directory (or, for Windows users who skipped step 4, the composers_assistant_nn_server.exe file). The server window will need to remain open for the scripts to work. (Note: The server is just a separate process that runs on your computer. It does NOT send any information over the internet.)

***STEP 7***: Within REAPER, add the "JS: Global Options for Composer's Assistant v2" fx your Monitor FX chain (view > Monitoring Effects). This fx will need to stay in your Monitor FX to have any effect. Add the "JS: Track-Specific Generation Options for Composer's Assistant v2" fx to any track that you want to set the infilling controls for before running one of the three REAPER scripts. ENJOY!

**NOTE**

The models in the release were trained only on public domain and permissively-licensed MIDI files. Please see the acknowledgements, disclaimer, and license in the download.

**HOW TO CITE COMPOSER'S ASSISTANT**

```
@inproceedings{ComposersAssistant,
title = {{Composer's Assistant: An Interactive Transformer for Multi-Track MIDI Infilling}},
author = {Martin E. Malandro},
booktitle = {{Proc. 24th Int. Society for Music Information Retrieval Conf.}},
year = 2023,
address = {Milan, Italy},
pages = {327--334},
}

@inproceedings{ComposersAssistant2,
title = {{Composer's Assistant 2: Interactive Multi-Track MIDI Infilling with Fine-Grained User Control}},
author = {Martin E. Malandro},
booktitle = {{Proc. 25th Int. Society for Music Information Retrieval Conf.}},
year = 2024,
address = {San Francisco, CA, USA},
pages = {438--445},
}
```
