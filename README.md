# tau identification with BDT 

XGBoost based on Boosted Decision Tree (BDT) to identify hadronic decays of tau

The result is published as a detector performance note (DPS) here:
https://cds.cern.ch/record/2727092?ln=en

#hadd all channel files together using 
`hadd -f TauData_Run2UL_full.root /vols/cms/dw515/outputs/SM/MVADM_UL/201*/*_{tt,mt,et}_*`

# reduce the size of this file by dropping events without hadronic taus usig:
`python TrimFile.py`

# run all following steps on batch asking for maximum memory


