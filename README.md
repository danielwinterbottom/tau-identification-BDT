# tau identification with BDT 

XGBoost based on Boosted Decision Tree (BDT) to identify hadronic decays of tau

The result is published as a detector performance note (DPS) here:
https://cds.cern.ch/record/2727092?ln=en

#hadd all channel files together using 
`hadd -f TauData_Run2UL_full.root /vols/cms/dw515/outputs/SM/MVADM_UL/201*/*_{tt,mt,et}_*`

# reduce the size of this file by dropping events without hadronic taus usig:
`python TrimFile.py`

# run all following steps on batch asking for maximum memory

# prepare dataframes for 1 prong:

`qsub -e /dev/null -o /dev/null -V -q hep.q -l h_rt=0:180:0 -l h_vmem=24G -cwd batch_MakeDataframe1prong.sh`

# prepare dataframes for 3 prong:

`qsub -e /dev/null -o /dev/null -V -q hep.q -l h_rt=0:180:0 -l h_vmem=24G -cwd batch_MakeDataframe1prong.sh`

# setup gpu support

# create virtual environment
`python -m pip install --user virtualenv`
`python -m virtualenv xgboost_gpu_env`

# activate virtual environment
`source xgboost_gpu_env/bin/activate`
# install xgboost with gpu support
`python -m pip install xgboost`
`python -m pip install xgboost --upgrade`
# deactivate virtual environment
`deactivate`

# to run training on gpu nodes

# train 1 prong:
`qsub -e /dev/null -o /dev/null -V -q gpu.q -l h_rt=24:0:0 -cwd batch_MultiClass1prong.sh`

# train 3 prong:
`qsub -e /dev/null -o /dev/null -V -q gpu.q -l h_rt=24:0:0 -cwd batch_MultiClass3prong.sh`

# to run on cpu's you need to change parameter 'tree_method' - can just comment it out or set it to auto
# you will also likely need to ask for 2 cpu cores to ensure memory is sufficient using:
`-pe hep.pe 2`
