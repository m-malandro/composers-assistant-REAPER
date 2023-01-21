# composers-assistant-REAPER
Welcome to the repository for Composer's Assistant for REAPER. If you're looking for the repository for the academic version of this project instead, click here: XXXXXXXXXXX

Installation is a bit more involved than your average REAPER script. If you want to know whether Composer's Assistant is right for you before attempting to install it, please watch this video for audio examples and to learn how it works: XXXXXXXXXXXX

**Installation instructions:**

First, if you don't already have python 3.6 or higher installed, install it and configure REAPER to see it. (In REAPER, go to Options > Preferences > Plug-Ins > ReaScript)

Second, install the following three libraries for your python installation: pytorch, transformers, and sentencepiece.

-Pytorch installation: Go to https://pytorch.org/ to get the command you need to run. You'll want the stable build for your OS, using the "Pip" package for Python (NOT Conda). If you have an NVIDIA card, it is recommended to install a CUDA version. Otherwise, install the CPU version. The CPU version will work. It's just slower. The command you use should start with something like "pip3 install torch..."

-Transformers installation: Just do ``pip install transformers``

-Sentencepiece installation: Just do ``pip install sentencepiece``

Third, download and unzip the release to your REAPER scripts folder. (To find your REAPER scripts folder, from within REAPER go to Options > Show REAPER resource path in explorer/finder..., and then open the folder called Scripts.) It is highly recommended that you unzip to a folder *within* your scripts folder instead of dumping everything directly into the scripts folder, but it's up to you.

From here, you load the scripts into REAPER in the usual way: In REAPER, go to Actions > Show action list..., then click on New action > Load ReaScript..., and then select all six scripts that start with "CA_". Ignore the other files. They are just helper files that these six scripts need to run.

**How to cite Composer's Assistant:**

Coming Soon.
