import uproot # can also use root_pandas or root_numpy
import numpy as np
import pandas as pd
import pickle
import time
import psutil
import os
from optparse import OptionParser
start_time = time.time()

parser = OptionParser()
parser.add_option("-d","--dir", dest="dir", type='string',default='./',
                  help="Specify the working directory")
(options, args) = parser.parse_args()
#directory = '/vols/cms/dw515/workareas/mvadm_ul/CMSSW_10_2_19/src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTauRun2/tau-identification-BDT/'
directory = options.dir

print("working")
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

variables = [          "gen_match_1", # gen match info (==5 for real tau_h)
                       "tau_decay_mode_1", # reco tau decay mode
                       "tauFlag1", # gen tau decay mode
                       "E_1","eta_1","pt_1", # tau variables
                       "Egamma1_1","Egamma2_1",# e/gamma candidate energy
                       "Epi0_1","Epi_1", # pi0 and p energies
                       "rho_dEta_1","rho_dphi_1",
                       "gammas_dEta_1","gammas_dphi_1",
                       "Mrho_1", # mass of rho (if DM=1, otherwise visible mass of tau)
                       "Mpi0_1",
                       "DeltaR2WRTtau_1",
                       "Mpi0_TwoHighGammas_1",
                       "Mrho_OneHighGammas_1",
                       "Mrho_TwoHighGammas_1",
                       "Mrho_subleadingGamma_1",
                       "strip_pt_1"
            ]

variables_2 = [        "gen_match_2", # gen match info (==5 for real tau_h)
                       "tau_decay_mode_2", # reco tau decay mode
                       "tauFlag2", # gen tau decay mode
                       "E_2","eta_2","pt_2", # tau variables
                       "Egamma1_2","Egamma2_2",# e/gamma candidate energy
                       "Epi0_2","Epi_2", # pi0 and p energies
                       "rho_dEta_2","rho_dphi_2",
                       "gammas_dEta_2","gammas_dphi_2",
                       "Mrho_2", # mass of rho (if DM=1, otherwise visible mass of tau)
                       "Mpi0_2",
                       "DeltaR2WRTtau_2",
                       "Mpi0_TwoHighGammas_2",
                       "Mrho_OneHighGammas_2",
                       "Mrho_TwoHighGammas_2",
                       "Mrho_subleadingGamma_2",
                       "strip_pt_2"
            ]



with uproot.open(directory+"/TauData_Run2UL.root") as file1:

  print('opened file')
  print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
  tree1 = file1["train_ntuple"] 
  print tree1

  print('loading variables for tau with index 1')
  print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
  df_1 = tree1.pandas.df(variables)
  print('finished loading variables for tau with index 1')
  print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
  print df_1[:10]
  print(df_1.memory_usage())

  df_1 = df_1[
    (df_1["tau_decay_mode_1"] <3)
    &(df_1["gen_match_1"] == 5)
    &(df_1["tauFlag1"] <3)
    &(df_1["tauFlag1"] >=0)
  ]
  df_1 = df_1.drop(["gen_match_1"], axis=1).reset_index(drop=True)
  
  print(df_1.memory_usage())

  # df_2 will be for the subleading0 tau ("*_2") in mutau input
  print('loading variables for tau with index 2')
  print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
  df_2 = tree1.pandas.df(variables_2)
  print('finished loading variables for tau with index 2')
  print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

  df_2 = df_2[
    (df_2["tau_decay_mode_2"] <3)
    &(df_2["gen_match_2"] == 5)
    &(df_2["tauFlag2"] <3)
    &(df_2["tauFlag2"] >=0)
  ]
  df_2 = df_2.drop(["gen_match_2"], axis=1).reset_index(drop=True)

  # define some new variables
  print('defining new variables for training') 
  print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
  
  print('defining variables for tau with index 1')
  print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
  # for df_1
  df_1.loc[:,"Egamma1_tau"] = df_1["Egamma1_1"] / df_1["E_1"]
  df_1 = df_1.drop(["Egamma1_1"], axis=1).reset_index(drop=True)
  df_1.loc[:,"Egamma2_tau"] = df_1["Egamma2_1"] / df_1["E_1"]
  df_1 = df_1.drop(["Egamma2_1"], axis=1).reset_index(drop=True)
  df_1.loc[:,"Epi_tau"]     = df_1["Epi_1"] / df_1["E_1"]
  df_1.loc[:,"rho_dEta_tau"] = df_1["rho_dEta_1"] * df_1["E_1"]
  df_1.loc[:,"rho_dphi_tau"] = df_1["rho_dphi_1"] * df_1["E_1"]
  df_1.loc[:,"gammas_dEta_tau"] = df_1["gammas_dEta_1"] * df_1["E_1"]
  df_1.loc[:,"gammas_dR_tau"]     = np.sqrt(df_1["gammas_dEta_1"]*df_1["gammas_dEta_1"] + df_1["gammas_dphi_1"]*df_1["gammas_dphi_1"])* df_1["E_1"]
  df_1 = df_1.drop(["gammas_dphi_1"], axis=1).reset_index(drop=True)
  df_1.loc[:,"DeltaR2WRTtau_tau"] = df_1["DeltaR2WRTtau_1"] * df_1["E_1"]* df_1["E_1"]
  df_1 = df_1.drop(["E_1"], axis=1).reset_index(drop=True)
  
  
  print('defining variables for tau with index 2')
  print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
  # now the same for df_2
  df_2.loc[:,"Egamma1_tau"] = df_2["Egamma1_2"] / df_2["E_2"]
  df_2 = df_2.drop(["Egamma1_2"], axis=1).reset_index(drop=True)
  df_2.loc[:,"Egamma2_tau"] = df_2["Egamma2_2"] / df_2["E_2"]
  df_2 = df_2.drop(["Egamma2_2"], axis=1).reset_index(drop=True)
  df_2.loc[:,"Epi_tau"]     = df_2["Epi_2"] / df_2["E_2"]
  df_2.loc[:,"rho_dEta_tau"] = df_2["rho_dEta_2"] * df_2["E_2"]
  df_2.loc[:,"rho_dphi_tau"] = df_2["rho_dphi_2"] * df_2["E_2"]
  df_2.loc[:,"gammas_dEta_tau"] = df_2["gammas_dEta_2"] * df_2["E_2"]
  df_2.loc[:,"gammas_dR_tau"]     = np.sqrt(df_2["gammas_dEta_2"]*df_2["gammas_dEta_2"] + df_2["gammas_dphi_2"]*df_2["gammas_dphi_2"])* df_2["E_2"]
  df_2 = df_2.drop(["gammas_dphi_2"], axis=1).reset_index(drop=True)
  df_2.loc[:,"DeltaR2WRTtau_tau"] = df_2["DeltaR2WRTtau_2"] * df_2["E_2"]* df_2["E_2"]
  df_2 = df_2.drop(["E_2"], axis=1).reset_index(drop=True)

  print('renaming variables')
  print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

  for key, values in df_1.iteritems():
      if "_1" in key:
          print(key)
          print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
          df_1.loc[:,key[:-2]] = df_1[key]
          df_1 = df_1.drop(key, axis=1).reset_index(drop=True)
  
  df_1.loc[:,"tauFlag"] = df_1["tauFlag1"]
  df_1 = df_1.drop("tauFlag1", axis=1).reset_index(drop=True)
  
  for key, values in df_2.iteritems():
      if "_2" in key:
          print(key)
          print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
          df_2.loc[:,key[:-2]] = df_2[key]
          df_2 = df_2.drop(key, axis=1).reset_index(drop=True)
  
  df_2.loc[:,"tauFlag"] = df_2["tauFlag2"]
  df_2 = df_2.drop("tauFlag2", axis=1).reset_index(drop=True)
  
print('closed file')
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

df_1 = pd.concat([df_1,df_2], ignore_index=True)
## concatinating uses too much memory so need to use this method:
## write df_1 content in file.csv
#df_1.to_csv('file.csv', index=False)
## append df_2 content to file.csv
#df_2.to_csv('file.csv', mode='a', index=False, header=False)
## free memory
#del df_1, df_2
#print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
#
#print ('opening csv files and creating concatinated dataframe')
## read all df_1 and df_2 contents
#df_1 = pd.read_csv(directory+'file.csv')

print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

# shuffle dataframe - this is needed as datasets mix DY and Higgs samples
df_1 = df_1.sample(frac=1).reset_index(drop=True)

print df_1[:10]

df_1=df_1.dropna().reset_index(drop=True)

df_1.to_csv('df_1prong_Run2UL.csv', index=False)

#with open (directory+"/df_1prong_Run2UL.pkl",'w') as f:
#    pickle.dump(df_1,f)
print 'dataframe saved!'

elapsed_time = time.time() - start_time
print "elapsed time=", elapsed_time

