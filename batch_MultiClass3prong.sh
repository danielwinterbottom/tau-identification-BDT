#!/bin/bash

DIR=$CMSSW_BASE/src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTauRun2/tau-identification-BDT
cd $DIR
pwd
export SCRAM_ARCH=slc6_amd64_gcc481
eval `scramv1 runtime -sh`

ulimit -c 0
ulimit -s unlimited

source xgboost_gpu_env/bin/activate

python -u MultiClass3prong.py --dir=$DIR &> MultiClass3prong.log 
