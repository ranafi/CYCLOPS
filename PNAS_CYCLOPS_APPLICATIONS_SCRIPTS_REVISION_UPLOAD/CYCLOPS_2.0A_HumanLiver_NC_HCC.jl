addprocs(5)
LOAD_PATH
LOAD_PATH=LOAD_PATH[1:2]
basedir=homedir();
netdir=string(basedir,"/Google Drive/PNAS_CYCLOPS_SCRIPTS/PNAS_CYCLOPS_PROGRAM_SCRIPTS_REVISION_UPLOAD");
cd(netdir);
push!(LOAD_PATH,netdir)

using StatsBase
using MultivariateStats
using Distributions
using PyPlot

using CYCLOPS_2a_AutoEncoderModule
using CYCLOPS_2a_PreNPostprocessModule
using CYCLOPS_2a_MCA
using CYCLOPS_2a_MultiCoreModule_Smooth
using CYCLOPS_2a_Seed


###################################################
#  Additional Recquired Functions #
###################################################
using Extra_HCC_Functions
using HypothesisTests

##########################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
# Main Program       ######################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################

indir=string(basedir,"/Documents/HCCWork_Nov2014/DataPartitions");
homologuedir1=string(basedir,"/Google Drive/BHTC_Homologues")
outdir=string(basedir,"/Google Drive/CYCLOPS_OUTPUT_FINAL_DISP/Liver/Nov2016_r6")

############################################################
Frac_Var=0.85 # Set Number of Dimensions of SVD to maintain this fraction of variance
DFrac_Var=0.03 # Set Number of Dimensions of SVD to so that incremetal fraction of variance of var is at least this much
N_trials =40  # Number of random initial conditions to try for each optimization
MaxSeeds = 10000

total_background_num=100; # Number of background runs for background bootstrap statistics
n_cores=5; # Number of machine cores
############################################################

cd(homologuedir1)
seed_homologues1=readcsv("LiverCyclerHomologues.csv");

homologue_symbol_list1=seed_homologues1[2:end,2];
############################################################
srand(123456);
############################################################

cd(indir);
fullnonseed_data_nml=readcsv("UnloggedNonTumorData.csv");
fullnonseed_data_hcc=readcsv("UnloggedTumorData.csv");


alldata_symbols=fullnonseed_data_nml[2:end , 2];
nprobes=size(alldata_symbols)[1]
cutrank=nprobes-MaxSeeds

Seed_MinMean_nml= (sort(vec(mean(float(fullnonseed_data_nml[2:end,4:end]),2))))[cutrank]
Seed_MinMean_hcc= (sort(vec(mean(float(fullnonseed_data_hcc[2:end,4:end]),2))))[cutrank]


srand(12345);

seed_probes_nml,seed_symbols_nml1, seed_data_nml1   =   getseed_woutputprobe(fullnonseed_data_nml,homologue_symbol_list1,.7,.14,Seed_MinMean_nml,.975);
newseed_probes1=seed_probes_nml

seed_probes_nml1,seed_symbols_nml1, seed_data_nml1  = getseed_winputprobe(fullnonseed_data_nml,newseed_probes1,.975);
seed_probes_hcc1,seed_symbols_hcc1, seed_data_hcc1  = getseed_winputprobe(fullnonseed_data_hcc,newseed_probes1,.975);

seed_data_dispersion_nml1                           = dispersion!(seed_data_nml1)
seed_data_dispersion_hcc1                           = dispersion!(seed_data_hcc1)

#######################################
# Get EigenGenes (and internally Eigenvectors) of NML data
#######################################
outs_nml1o,norm_seed_data_nml1o                        =   GetEigenGenes(seed_data_dispersion_nml1,Frac_Var,DFrac_Var,50)


#######################################
# Seperately get Eigenvectors of NML data and project both NML data and HCC data onto these
# The rescale Projected data so NML data has variance of 1 (variance of original Eigengene data)
#######################################
outs_nml1, norm_seed_data_nml1, norm_seed_data_hcc1     =   GetEigenGenes2(seed_data_dispersion_nml1,seed_data_dispersion_hcc1,Frac_Var,outs_nml1o)

rescale1                =norm_seed_data_nml1o[1,1]/norm_seed_data_nml1[1,1]

norm_seed_data_nml1     =rescale1*norm_seed_data_nml1
norm_seed_data_hcc1     =rescale1*norm_seed_data_hcc1

outs_hcc1=outs_nml1

nml1 =@spawn CYCLOPS_Order(outs_nml1,norm_seed_data_nml1,N_trials); 
hcc1= @spawn CYCLOPS_Order(outs_hcc1,norm_seed_data_hcc1,N_trials); 

estimated_phaselist_nml1,bestnet_nml1,global_var_metrics_nml1=fetch(nml1);
estimated_phaselist_hcc1,bestnet_hcc1,global_var_metrics_hcc1=fetch(hcc1);

subplot(1,2,1)
plt.hist(estimated_phaselist_nml1,alpha=.25)
plt.hist(mod(estimated_phaselist_hcc1,2*pi),alpha=.25)

cd(outdir);

ylabp=[0,pi/2,pi,3*pi/2,2*pi]
ylabs=[0, "","π","","2π"]
xlabp=[0,pi/2,pi,3*pi/2,2*pi]
xlabs=[0, "","π","","2π"]

cosinor_nml1=Compile_MultiCore_Cosinor_Statistics(fullnonseed_data_nml,estimated_phaselist_nml1,4,48)
cosinor_hcc1=Compile_MultiCore_Cosinor_Statistics(fullnonseed_data_hcc,estimated_phaselist_hcc1,4,48)

############################################################
function Filter_Cosinor_Output(cosdata::Array{Any,2},pval,rsq,ptt) 
    significant_data=cosdata[[true,cosdata[2:end,5].<pval],:];
    phys_sig_data=significant_data[[true,significant_data[2:end,11].> ptt],:];
    strong_data=phys_sig_data[[true,phys_sig_data[2:end,10].> rsq],:];
    strong_data
end
############################################################
############################################################
function Circular_Mean(phases::Array{Float64,1})
  sinterm=sum(sin(phases))
  costerm=sum(cos(phases))
  atan2(sinterm,costerm)
end 

############################################################
# Align to EBOX phase (par BZip)
############################################################
sig_nml1=Filter_Cosinor_Output(cosinor_nml1,.05,0,1.66)
sig_hcc1=Filter_Cosinor_Output(cosinor_hcc1,.05,0,1.66)

eboxgenes=["DBP","HLF","TEF"]

eboxphases_nml=cosinor_nml1[[findin(cosinor_nml1[:,2],eboxgenes)],:]
nml_criteria=((eboxphases_nml[:,4].<.05) & (eboxphases_nml[:,11].>1.25) & (eboxphases_nml[:,9].>100))

eboxphases_nml=eboxphases_nml[findin(nml_criteria,true),:]

eboxphases=float(eboxphases_nml[:,6])

eboxphase=Circular_Mean(eboxphases)

estimated_phaselist_nml1_s=mod(estimated_phaselist_nml1 .- eboxphase+(pi),2*pi)  ##ebox genes in mouse lung peak on average CT 11.5 

Nday=length(findin((estimated_phaselist_nml1_s .>pi),true))
Nnight=length(findin((estimated_phaselist_nml1_s .<pi),true))

if (Nday<Nnight)
     estimated_phaselist_nml1_s=mod(2*pi-estimated_phaselist_nml1_s,2*pi)
end

############################################################
# Align to HCC to NML 
############################################################

cosinor_nml1_s=Compile_MultiCore_Cosinor_Statistics(fullnonseed_data_nml,estimated_phaselist_nml1_s,4,48)
sig_nml1_s=Filter_Cosinor_Output(cosinor_nml1_s,.05,0,1.66)

common_n1_s_h1=intersect(sig_nml1_s[2:end,1],sig_hcc1[2:end,1])
rows_nml1_s=findin(sig_nml1_s[:,1],common_n1_s_h1)
rows_hcc1=findin(sig_hcc1[:,1],common_n1_s_h1)

bestacrophaselist1,estimated_phaselist_hcc1_s=best_shift_cos2(sig_hcc1[rows_hcc1,6],sig_nml1_s[rows_nml1_s,6],estimated_phaselist_hcc1,"radians")

close()
plt.hist(estimated_phaselist_nml1_s,alpha=.25)
plt.hist(estimated_phaselist_hcc1_s,alpha=.25)


############################################################
# Core in both HCC and NML
#########################################################
cosinor_nml1_s=Compile_MultiCore_Cosinor_Statistics(fullnonseed_data_nml,estimated_phaselist_nml1_s,4,48)
cosinor_hcc1_s=Compile_MultiCore_Cosinor_Statistics(fullnonseed_data_hcc,estimated_phaselist_hcc1_s,4,48)

sig_nml1_s=Filter_Cosinor_Output(cosinor_nml1_s,.05,0,1.66)
sig_hcc1_s=Filter_Cosinor_Output(cosinor_hcc1_s,.05,0,1.66)

common_n1_s_h1_s=intersect(sig_nml1_s[2:end,1],sig_hcc1_s[2:end,1])
rows_nml1_s=findin(sig_nml1_s[:,1],common_n1_s_h1_s)
rows_hcc1_s=findin(sig_hcc1_s[:,1],common_n1_s_h1_s)

close()
clockgenes=["CLOCK","ARNTL","PER1","PER2","PER3","CRY1","CRY2","RORA","RORB","NR1D1","NR1D2","FBXL3","C1orf51","CHRONO","CIART","DBP","TEF","HLF"]
clockrows=findin(cosinor_nml1_s[:,2],clockgenes)
clockrows=clockrows[[4,13,5,14,6,20,18,9,21]]

clockdata_nml=float(fullnonseed_data_nml[clockrows,4:end])
clockdata_hcc=float(fullnonseed_data_hcc[clockrows,4:end])

clockannotations=alldata_symbols[clockrows-1]

clockcosinor_nml1_s=cosinor_nml1_s[clockrows,:]
clockcosinor_hcc1_s=cosinor_hcc1_s[clockrows,:]

xlabp=[0,pi/2,pi,3*pi/2,2*pi]
xlabs=[0, "","π","","2π"]
ylabp=[]
ylabs=[]

for n in 1:9
          m1=2n-1
          m2=2n

          subplot(3,6,m1)

          ymean1=mean(clockdata_nml[n,:])
          ymax1=minimum([ymean1+3*std(clockdata_nml[n,:]),maximum(clockdata_nml[n,:])])
          ymin1=maximum([0,ymean1-3*std(clockdata_nml[n,:])])

          ymaxstar=int(100*div(ymax1,100))
          yminstar=int(100*div(ymin1,100))
          ymedstar=int((yminstar+ymaxstar)/2)
          
          ylabp=[yminstar,ymedstar,ymaxstar]
          ylabs=[yminstar,ymedstar,ymaxstar]

          axis([0,2*pi,ymin1,ymax1])
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs, fontsize=8)

          scatter(estimated_phaselist_nml1_s,clockdata_nml[n,:],alpha=.75,s=7,color="Black")

          title(clockannotations[n], fontsize=18)
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs)

          PrbP,PrbPhase,PrbAmp,PrbMean=clockcosinor_nml1_s[n,[4,6,7,8]]
          sest=linspace(0,2*pi,100)
          synth=PrbAmp*cos(sest-PrbPhase)+PrbMean
          if PrbP<0.05
            plot(sest,synth,"r-",lw=2)
          end

          subplot(3,6,m2)
          ymean2=mean(clockdata_hcc[n,:])
          ymax2=minimum([ymean2+3*std(clockdata_hcc[n,:]),maximum(clockdata_hcc[n,:])])
          ymin2=maximum([0,ymean2-3*std(clockdata_hcc[n,:])])

          ymaxstar=int(100*div(ymax2,100))
          yminstar=int(100*div(ymin2,100))
          ymedstar=int((yminstar+ymaxstar)/2)
          
          ylabp=[yminstar,ymedstar,ymaxstar]
          ylabs=[yminstar,ymedstar,ymaxstar]


          axis([0,2*pi,ymin2,ymax2])

          scatter(estimated_phaselist_hcc1_s,clockdata_hcc[n,:],alpha=.75,s=7,color="DarkRed")

          title(clockannotations[n], fontsize=18)
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs, fontsize=8)

          PrbP,PrbPhase,PrbAmp,PrbMean=clockcosinor_hcc1_s[n,[4,6,7,8]]
          sest=linspace(0,2*pi,100)
          synth=PrbAmp*cos(sest-PrbPhase)+PrbMean
          if PrbP<0.05
            plot(sest,synth,"r-",lw=2)
          end
          
end


############################################################
# Core in  NML
#########################################################
close()
for n in 1:9
          m1=2n-1
          m2=2n
          subplot(3,3,n)

          ymean1=mean(clockdata_nml[n,:])
          ymax1=minimum([ymean1+3*std(clockdata_nml[n,:]),maximum(clockdata_nml[n,:])])
          ymin1=maximum([0,ymean1-3*std(clockdata_nml[n,:])])

          ymaxstar=int(100*div(ymax1,100))
          yminstar=int(100*div(ymin1,100))
          ymedstar=int((yminstar+ymaxstar)/2)
          
          ylabp=[yminstar,ymedstar,ymaxstar]
          ylabs=[yminstar,ymedstar,ymaxstar]

          axis([0,2*pi,ymin1,ymax1])
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs, fontsize=10)

          scatter(estimated_phaselist_nml1_s,clockdata_nml[n,:],alpha=.75,s=7,color="Black")

          title(clockannotations[n], fontsize=18)
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs)

          PrbP,PrbPhase,PrbAmp,PrbMean=clockcosinor_nml1_s[n,[4,6,7,8]]
          sest=linspace(0,2*pi,100)
          synth=PrbAmp*cos(sest-PrbPhase)+PrbMean
          if PrbP<0.05
            plot(sest,synth,"r-",lw=2)
          end

end

############################################################
# Drug targets in NML
#########################################################

close()
clockgenes=["PPARA","XDH","PDE4A","PDE4B","PDE5A","DDC","AGTR1"]
clockrows=findin(cosinor_nml1_s[:,2],clockgenes)
clockrows=clockrows[[8,6,2]]

clockdata_nml=float(fullnonseed_data_nml[clockrows,4:end])
clockannotations=alldata_symbols[clockrows-1]

clockcosinor_nml1_s=cosinor_nml1_s[clockrows,:]
clockcosinor_hcc1_s=cosinor_hcc1_s[clockrows,:]

xlabp=[0,pi/2,pi,3*pi/2,2*pi]
xlabs=[0, "","π","","2π"]
ylabp=[]
ylabs=[]

for n in 1:3
          m1=2n-1
          m2=2n
          subplot(3,1,n)

          ymean1=mean(clockdata_nml[n,:])
          ymax1=minimum([ymean1+3*std(clockdata_nml[n,:]),maximum(clockdata_nml[n,:])])
          ymin1=maximum([0,ymean1-3*std(clockdata_nml[n,:])])

          ymaxstar=int(100*div(ymax1,100))
          yminstar=int(100*div(ymin1,100))
          ymedstar=int((yminstar+ymaxstar)/2)
          
          ylabp=[yminstar,ymedstar,ymaxstar]
          ylabs=[yminstar,ymedstar,ymaxstar]

          axis([0,2*pi,ymin1,ymax1])
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs, fontsize=10)

          scatter(estimated_phaselist_nml1_s,clockdata_nml[n,:],alpha=.75,s=7,color="Black")

          title(clockannotations[n], fontsize=18)
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs)

          PrbP,PrbPhase,PrbAmp,PrbMean=clockcosinor_nml1_s[n,[4,6,7,8]]
          sest=linspace(0,2*pi,100)
          synth=PrbAmp*cos(sest-PrbPhase)+PrbMean
          if PrbP<0.05
            plot(sest,synth,"r-",lw=2)
          end

end

############################################################
# Outputs in both HCC and NML
#########################################################
close()
clockgenes=["ARNTL2","MYC","TNFRSF12A","TNFRSF10A","TNFRSF10B","SOCS1","SOCS3","STAT3","PTPN","PHGDH","PSAT1","SHMT1","PIM1","PDXK","ODC1","MTHFD2","CYP2C8"]
clockrows=findin(cosinor_nml1_s[:,2],clockgenes)
clockrows=clockrows[[10,8,22,20,12,13,5,19,16]]

clockdata_nml=float(fullnonseed_data_nml[clockrows,4:end])
clockdata_hcc=float(fullnonseed_data_hcc[clockrows,4:end])

clockannotations=alldata_symbols[clockrows-1]

clockcosinor_nml1_s=cosinor_nml1_s[clockrows,:]
clockcosinor_hcc1_s=cosinor_hcc1_s[clockrows,:]

xlabp=[0,pi/2,pi,3*pi/2,2*pi]
xlabs=[0, "","π","","2π"]
ylabp=[]
ylabs=[]

for n in 1:9
          m1=2n-1
          m2=2n

          subplot(3,6,m1)

          ymean1=mean(clockdata_nml[n,:])
          ymax1=minimum([ymean1+3*std(clockdata_nml[n,:]),maximum(clockdata_nml[n,:])])
          ymin1=maximum([0,ymean1-3*std(clockdata_nml[n,:])])

          ymean2=mean(clockdata_hcc[n,:])
          ymax2=minimum([ymean2+3*std(clockdata_hcc[n,:]),maximum(clockdata_hcc[n,:])])
          ymin2=maximum([0,ymean2-3*std(clockdata_hcc[n,:])])

          ymin=minimum([ymin1,ymin2])
          ymax=maximum([ymax1,ymax2])

          ymaxstar=int(100*div(ymax,100))
          yminstar=int(100*div(ymin,100))
          ymedstar=int((yminstar+ymaxstar)/2)
          
          ylabp=[yminstar,ymedstar,ymaxstar]
          ylabs=[yminstar,ymedstar,ymaxstar]


          axis([0,2*pi,ymin,ymax])
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs, fontsize=8)

          scatter(estimated_phaselist_nml1_s,clockdata_nml[n,:],alpha=.75,s=7,color="Black")

          title(clockannotations[n], fontsize=14)
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs)

          PrbP,PrbPhase,PrbAmp,PrbMean=clockcosinor_nml1_s[n,[4,6,7,8]]
          sest=linspace(0,2*pi,100)
          synth=PrbAmp*cos(sest-PrbPhase)+PrbMean
          if PrbP<0.05
            plot(sest,synth,"r-",lw=2)
          end

          subplot(3,6,m2)
          
          ylabp=[yminstar,ymedstar,ymaxstar]
          ylabs=["","",]


          axis([0,2*pi,ymin,ymax])

          scatter(estimated_phaselist_hcc1_s,clockdata_hcc[n,:],alpha=.75,s=7,color="DarkRed")

          title(clockannotations[n], fontsize=14)
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs)

          PrbP,PrbPhase,PrbAmp,PrbMean=clockcosinor_hcc1_s[n,[4,6,7,8]]
          sest=linspace(0,2*pi,100)
          synth=PrbAmp*cos(sest-PrbPhase)+PrbMean
          if PrbP<0.05
            plot(sest,synth,"r-",lw=2)
          end
          
end



############################################################
# Outputs in both HCC and NML v2
#########################################################
close()
clockgenes=["ARNTL2","TNFRSF12A","TNFRSF10B","SOCS1","SOCS3","STAT3","PHGDH","PSAT1","PIM1","YAP1","HDAC8","CDKN1A","EGLN1"]
clockrows=findin(cosinor_nml1_s[:,2],clockgenes)
clockrows=clockrows[[9,2,17,16,13,20,5,15,14]]

clockdata_nml=float(fullnonseed_data_nml[clockrows,4:end])
clockdata_hcc=float(fullnonseed_data_hcc[clockrows,4:end])

clockannotations=alldata_symbols[clockrows-1]

clockcosinor_nml1_s=cosinor_nml1_s[clockrows,:]
clockcosinor_hcc1_s=cosinor_hcc1_s[clockrows,:]

xlabp=[0,pi/2,pi,3*pi/2,2*pi]
xlabs=[0, "","π","","2π"]
ylabp=[]
ylabs=[]

for n in 1:9
          m1=2n-1
          m2=2n

          subplot(3,6,m1)

          ymean1=mean(clockdata_nml[n,:])
          ymax1=minimum([ymean1+3*std(clockdata_nml[n,:]),maximum(clockdata_nml[n,:])])
          ymin1=maximum([0,ymean1-3*std(clockdata_nml[n,:])])

          ymean2=mean(clockdata_hcc[n,:])
          ymax2=minimum([ymean2+3*std(clockdata_hcc[n,:]),maximum(clockdata_hcc[n,:])])
          ymin2=maximum([0,ymean2-3*std(clockdata_hcc[n,:])])

          ymin=minimum([ymin1,ymin2])
          ymax=maximum([ymax1,ymax2])

          ymaxstar=int(100*div(ymax,100))
          yminstar=int(100*div(ymin,100))
          ymedstar=int((yminstar+ymaxstar)/2)
          
          ylabp=[yminstar,ymedstar,ymaxstar]
          ylabs=[yminstar,ymedstar,ymaxstar]


          axis([0,2*pi,ymin,ymax])
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs, fontsize=8)

          scatter(estimated_phaselist_nml1_s,clockdata_nml[n,:],alpha=.75,s=7,color="Black")

          title(clockannotations[n], fontsize=14)
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs)

          PrbP,PrbPhase,PrbAmp,PrbMean=clockcosinor_nml1_s[n,[4,6,7,8]]
          sest=linspace(0,2*pi,100)
          synth=PrbAmp*cos(sest-PrbPhase)+PrbMean
          if PrbP<0.05
            plot(sest,synth,"r-",lw=2)
          end

          subplot(3,6,m2)
          
          ylabp=[yminstar,ymedstar,ymaxstar]
          ylabs=["","",]


          axis([0,2*pi,ymin,ymax])

          scatter(estimated_phaselist_hcc1_s,clockdata_hcc[n,:],alpha=.75,s=7,color="DarkRed")

          title(clockannotations[n], fontsize=14)
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs)

          PrbP,PrbPhase,PrbAmp,PrbMean=clockcosinor_hcc1_s[n,[4,6,7,8]]
          sest=linspace(0,2*pi,100)
          synth=PrbAmp*cos(sest-PrbPhase)+PrbMean
          if PrbP<0.05
            plot(sest,synth,"r-",lw=2)
          end
          
end


###############################################################################
# SLC2A2 Plot
###############################################################################

clockgenes=["SLC2A2"]
clockrows=findin(cosinor_nml1_s[:,2],clockgenes)

clockdata_nml=float(fullnonseed_data_nml[clockrows,4:end])
clockannotations=alldata_symbols[clockrows-1]

clockcosinor_nml1_s=cosinor_nml1_s[clockrows,:]

n=1
          
ymean=mean(clockdata_nml[n,:])
ymax=minimum([ymean+3*std(clockdata_nml[n,:]),maximum(clockdata_nml[n,:])])
ymin=maximum([0,ymean-3*std(clockdata_nml[n,:])])
          
ymaxstar=int(100*div(ymax,100))
yminstar=int(100*div(ymin,100))
ymedstar=int((yminstar+ymaxstar)/2)
          
ylabp=[yminstar,ymedstar,ymaxstar]
ylabs=["","",""]
          
axis([0,2*pi,ymin,ymax])
xticks(xlabp, xlabs, fontsize=14)
yticks(ylabp, ylabs, fontsize=14)

scatter(estimated_phaselist_nml1_s,clockdata_nml[n,:],alpha=.75,s=25,color="Black")

PrbP,PrbPhase,PrbAmp,PrbMean=clockcosinor_nml1_s[n,[4,6,7,8]]
sest=linspace(0,2*pi,100)
synth=PrbAmp*cos(sest-PrbPhase)+PrbMean
if PrbP<0.05
     plot(sest,synth,"r-",lw=2)
end

###############################################################################
# Begin Quantitative/Global Comparisons of HCC and NC results
###############################################################################

phys_sig_nml=Filter_Cosinor_Output(cosinor_nml1_s,.05,0,2);
phys_sig_hcc=Filter_Cosinor_Output(cosinor_hcc1_s,.05,0,2);

phys_sig_nml_probes=phys_sig_nml[2:end,1];
phys_sig_hcc_probes=phys_sig_hcc[2:end,1];

phys_sig_nml_genes=union(phys_sig_nml[2:end,2],[]);
phys_sig_hcc_genes=union(phys_sig_hcc[2:end,2],[]);


size(phys_sig_nml_genes)
size(phys_sig_hcc_genes)

size(intersect(phys_sig_nml_genes,phys_sig_hcc_genes))

nml_probes=phys_sig_nml_probes
nml_only_probes=setdiff(phys_sig_nml_probes,phys_sig_hcc_probes)
both_probes=intersect(phys_sig_nml_probes,phys_sig_hcc_probes)

nml_rows=findin(fullnonseed_data_nml[:,1],nml_probes)
nml_only_rows=findin(fullnonseed_data_nml[:,1],nml_only_probes)
both_rows=findin(fullnonseed_data_nml[:,1],both_probes)



nml1_namp=float(cosinor_nml1_s[nml_rows,7] ./ cosinor_nml1_s[nml_rows,8])
hcc1_namp=float(cosinor_hcc1_s[nml_rows,7] ./ cosinor_hcc1_s[nml_rows,8])
close()

###############################################################################
# Histogram of cycling/amplitude changes with HCC
###############################################################################
close()
#subplot(1,2,1)
plt.hist(log(hcc1_namp./nml1_namp),normed=1,bins=linspace(-4,4,40))
title("Amplitude change with HCC", fontsize=20)
xlabel("Ln (HCC/Non-Cancerous Amplitude Ratio)", fontsize=14)
ylabel("Frequency", fontsize=14)


###############################################################################
# Nested Model to Assess Significance of difference between NML and HCC cyclers
# Identify genes that are (1) only cyclers in nml ("nml_omly")(2) significantly better fit by nested model and (3) sufficiently reduced amplitude
###############################################################################
subdata_nml=fullnonseed_data_nml[[1,nml_only_rows],:]
subdata_hcc=fullnonseed_data_hcc[[1,nml_only_rows],:]
amp_change_out=["Probe","Symbol","Entrez","Circ Change pval","NML_amp","TUMOR_amp","NML_acrophase","TUMOR_acrophase","Mann pval"]'
strict_amp_change_out=["Probe","Symbol","Entrez","Circ Change pval","NML_amp","TUMOR_amp","NML_acrophase","TUMOR_acrophase","Mann pval"]'

out1=[]
out2=[]
out3=[]
ncheck=size(subdata_nml)[1]
cut=.05/ncheck
c=0

for n in 2:ncheck
  out1=subdata_hcc[n,1:3]
  out2=Tumor_Compare2(float(subdata_nml[n,4:end])',float(subdata_hcc[n,4:end])',estimated_phaselist_nml1_s,estimated_phaselist_hcc1_s)
  out3=pvalue(MannWhitneyUTest(vec(float(subdata_nml[n,4:end]')),vec(float(subdata_hcc[n,4:end])')))
  if out2[1]<cut
    println(n)
    if out2[3]/out2[2] <.5
         amp_change_out=vcat(amp_change_out,[out1',out2,out3]')
    end
    if out2[3]/out2[2] <.25
         strict_amp_change_out=vcat(strict_amp_change_out,[out1',out2,out3]')
    end

  end  
end


###############################################################################
#For all transcripts cycling in the NML tissue - compare p value for change in mean vs p value for change in cycling
#At same time prepare list for pre-ranked GSEA-based on shcange in amplitude
###############################################################################
subdata_nml=fullnonseed_data_nml[[1,nml_rows],:]
subdata_hcc=fullnonseed_data_hcc[[1,nml_rows],:]
out=["Probe","Symbol","Entrez","Circ Change pval","NML_amp","TUMOR_amp","NML_acrophase","TUMOR_acrophase","Mann pval","Amp_Ratio"]'

out1=[]
out2=[]
out3=[]
out4=[]
ncheck=size(subdata_nml)[1]
cut=.05/ncheck
c=0

for n in 2:ncheck
  out1=subdata_hcc[n,1:3]
  out2=Tumor_Compare2(float(subdata_nml[n,4:end])',float(subdata_hcc[n,4:end])',estimated_phaselist_nml1_s,estimated_phaselist_hcc1_s)
  out3=pvalue(MannWhitneyUTest(vec(float(subdata_nml[n,4:end]')),vec(float(subdata_hcc[n,4:end])')))
  out4=[out2[3]/out2[2]]
  out=vcat(out,[out1',out2,out3,out4]')
end

compile_out1=["Symbol","Entrez","Amp_Ratio"]'
compile_out2=["Symbol","Entrez","Amp_Ratio"]'
compile_out3=["Symbol","Entrez","Amp_Ratio"]'
compile_out4=["Symbol","Entrez","Amp_Ratio"]'
outgenes=union(out[2:end,2],[])
outgenes=outgenes[outgenes .!=""]

#########
# if a gene is represented by more than 1 probe - find the average amp change for that gene given all sig probes
##########
for gene in outgenes
  rows=findin(out[:,2],[gene])
  entrez=out[rows[1],3]
  avg_amp_ratio1=mean(float(out[rows,end]))
  avg_amp_ratio2=exp(mean(log(float(out[rows,end]))))
  small_amp_ratio=minimum(float(out[rows,end]))
  big_amp_ratio=maximum(float(out[rows,end]))
  newrow1=[gene,entrez,avg_amp_ratio1]'
  newrow2=[gene,entrez,avg_amp_ratio2]'
  newrow3=[gene,entrez,small_amp_ratio]'
  newrow4=[gene,entrez,big_amp_ratio]'
 
  compile_out1=vcat(compile_out1,newrow1)
  compile_out2=vcat(compile_out2,newrow2)
  compile_out3=vcat(compile_out3,newrow3)
  compile_out4=vcat(compile_out4,newrow4)

end


close()
scatter(log(10,float(out[2:end,9])),log(10,float(out[2:end,4])),alpha=.5,s=5)
xlabel("Log (Mann Whitney p value)", fontsize=14)
ylabel("Log (Circadian Change p value)", fontsize=14)
###############################################################################
sig_amp_genelist=amp_change_out[2:end,2]
sig_amp_genelist=union(sig_amp_genelist,[])
sig_amp_genelist=sig_amp_genelist[sig_amp_genelist .!=""]
###############################################################################
sig_amp_entrezlist=amp_change_out[2:end,3]
sig_amp_entrezlist=union(sig_amp_entrezlist,[])
sig_amp_entrezlist=sig_amp_entrezlist[sig_amp_entrezlist .!=""]
sig_amp_entrezlist=sig_amp_entrezlist[sig_amp_entrezlist .!="NA"]
sig_amp_entrezlist=int(sig_amp_entrezlist)
###############################################################################
strict_sig_amp_genelist=strict_amp_change_out[2:end,2]
strict_sig_amp_genelist=union(strict_sig_amp_genelist,[])
strict_sig_amp_genelist=strict_sig_amp_genelist[strict_sig_amp_genelist .!=""]
###############################################################################
cd(outdir)
writecsv("DAVID_Loss_of_Cycling_HalfAmp.csv",sig_amp_genelist)
writecsv("DAVID_Loss_of_Cycling_HalfAmp_Entrez.csv",sig_amp_entrezlist)

writecsv("DAVID_Strict_Loss_of_Cycling_QuarterAmp.csv",strict_sig_amp_genelist)
writecsv("All_normal_cyclers.csv",out)
writecsv("GSEAReady_AvgLogAmpChange_NormalCyclers.csv",compile_out2)
writecsv("GSEAReady_AvgAmpChange_NormalCyclers.csv",compile_out1)
writecsv("GSEAReady_MaxAmpChange_NormalCyclers.csv",compile_out3)
writecsv("GSEAReady_MinAmpChange_NormalCyclers.csv",compile_out4)

writecsv("COSINOR_NC_Liver.csv",cosinor_nml1_s)
writecsv("COSINOR_HCC_Liver.csv",cosinor_nml1_s)
writecsv("NML_Phaselist.csv",estimated_phaselist_nml1_s)
writecsv("HCC_Phaselist.csv",estimated_phaselist_hcc1_s)
