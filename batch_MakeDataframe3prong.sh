#!/bin/bash

DIR=$CMSSW_BASE/src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTauRun2/tau-identification-BDT
cd $DIR
pwd
export SCRAM_ARCH=slc6_amd64_gcc481
eval `scramv1 runtime -sh`

ulimit -c 0
ulimit -s unlimited


python -u MakeDataframe3prong.py --dir=$DIR &> MakeDataframe3prong.log 
