import ROOT

fout = ROOT.TFile('TauData_Run2UL.root','RECREATE')
f1 = ROOT.TFile('tau-identification-BDT/TauData_Run2UL_full.root')

t1 = f1.Get('train_ntuple')
fout.cd()
tout1 = t1.CopyTree("gen_match_1==5||gen_match_2==5")

tout1.Write()

f1.Close()
fout.Close()

