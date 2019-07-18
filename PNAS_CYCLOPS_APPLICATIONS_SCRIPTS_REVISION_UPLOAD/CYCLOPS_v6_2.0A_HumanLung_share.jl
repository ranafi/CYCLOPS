addprocs(5)
################
#My local machine has 6 processors
#CYCLOPS is written to use them
#################
@everywhere basedir=homedir();
@everywhere netdir=string(basedir,"/Desktop/CYCLOPS_UPDATE_SHARE");
@everywhere cd(netdir);


using StatsBase
using MultivariateStats
using Distributions

@everywhere include("CYCLOPS_v6_2a_AutoEncoderModule_multi.jl")
@everywhere include("CYCLOPS_v6_2a_seed.jl")
@everywhere include("CYCLOPS_v6_2a_PreNPostprocessModule.jl")
@everywhere include("CYCLOPS_v6_2a_CircularStats_U.jl")
@everywhere include("CYCLOPS_v6_2a_MultiCoreModule_Smooth.jl")


using CYCLOPS_v6_2a_AutoEncoderModule_multi
using CYCLOPS_v6_2a_Seed
using CYCLOPS_v6_2a_PreNPostprocessModule
using CYCLOPS_v6_2a_CircularStats_U
using CYCLOPS_v6_2a_MultiCoreModule_Smooth

 
using PyPlot
#indirlaval=string(basedir,"/Documents/LungWork_March2015/LAVAL/UsableData");
#indirgrng=string(basedir,"/Documents/LungWork_March2015/GRNG/UsableData");
#homologuedir=string(basedir,"/Google Drive/BHTC_Homologues")

indirlaval=netdir;
indirgrng=netdir;
homologuedir=netdir;

############################################################
cd(indirlaval);
fullnonseed_data_laval=readcsv("AnnotatedLAVALData_March3_2015.csv");

cd(indirgrng);
fullnonseed_data_grng=readcsv("AnnotatedGRNGData_March3_2015.csv");

cd(homologuedir)
seed_homologues=readcsv("LungCyclerHomologues.csv");
homologue_symbol_list=seed_homologues[2:end,2];

############################################################
Frac_Var=0.85 # Set Number of Dimensions of SVD to maintain this fraction of variance
DFrac_Var=0.03 # Set Number of Dimensions of SVD to so that incremetal fraction of variance of var is at least this much
N_trials =20  # Number of random initial conditions to try for each optimization
MaxSeeds = 10000

total_background_num=20; # Number of background runs for global background refrence statistics
n_cores=5; # Number of machine cores
############################################################

alldata_data_l=fullnonseed_data_laval[2:end , 4:end];
alldata_data_l=Array{Float64}(alldata_data_l); 

alldata_data_g=fullnonseed_data_grng[2:end , 4:end];
alldata_data_g=Array{Float64}(alldata_data_g); 

alldata_symbols_g=fullnonseed_data_grng[2:end , 2];
nprobes=size(alldata_data_l)[1]
cutrank=nprobes-MaxSeeds

Seed_MinMean_l= (sort(vec(mean(alldata_data_l,2))))[cutrank]
Seed_MinMean_g= (sort(vec(mean(alldata_data_g,2))))[cutrank]

srand(123456);

seed_symbols_l, seed_data_l      =   getseed(fullnonseed_data_laval,homologue_symbol_list,.7,.14,Seed_MinMean_l,.975);
seed_data_dispersion_l           =   dispersion!(seed_data_l)
outs_l, norm_seed_data_l			   =	  GetEigenGenes(seed_data_dispersion_l,Frac_Var,DFrac_Var,30)

seed_symbols_g, seed_data_g      =   getseed(fullnonseed_data_grng,homologue_symbol_list,.7,.14,Seed_MinMean_g,.975);
seed_data_dispersion_g           =   dispersion!(seed_data_g)
outs_g, norm_seed_data_g				 =	 GetEigenGenes(seed_data_dispersion_g,Frac_Var,DFrac_Var,30)

#################
a_g =CYCLOPS_Order_multicore(outs_g,norm_seed_data_g,N_trials);  
a_l =CYCLOPS_Order_multicore(outs_l,norm_seed_data_l,N_trials);

#a_g =CYCLOPS_Order(outs_g,norm_seed_data_g,N_trials);  
#a_l =CYCLOPS_Order(outs_l,norm_seed_data_l,N_trials);

estimated_phaselist_g,bestnet_g,global_var_metrics_g=a_g;
estimated_phaselist_l,bestnet_l,global_var_metrics_l=a_l;


global_smooth_metrics_g            = smoothness_measures(seed_data_dispersion_g,norm_seed_data_g,estimated_phaselist_g)
global_metrics_g                   = global_var_metrics_g

global_smooth_metrics_l            = smoothness_measures(seed_data_dispersion_l,norm_seed_data_l,estimated_phaselist_l)
global_metrics_l                   = global_var_metrics_l

#pvals_g=multicore_backgroundstatistics_global_eigen(seed_data_dispersion_g,outs_g,N_trials,total_background_num,global_metrics_g)
#pvals_l=multicore_backgroundstatistics_global_eigen(seed_data_dispersion_l,outs_l,N_trials,total_background_num,global_metrics_l)



outdir=string(basedir,"/Google Drive/CYCLOPS_OUTPUT_FINAL_DISP/Lung/Nov2016_r1")
cd(outdir);

############################################################
function Filter_Cosinor_Output(cosdata::Array{Any,2},pval,rsq,ptt) 
    significant_data=cosdata[append!([1],1 + findin(cosdata[2:end,5] .< pval,true)),:];
    phys_sig_data=significant_data[append!([1],1 + findin(significant_data[2:end,11] .> ptt,true)),:];
    strong_data=phys_sig_data[append!([1],1 + findin(phys_sig_data[2:end,10].> rsq,true)),:];
    strong_data
end
############################################################

estimated_phaselist_g=mod.(estimated_phaselist_g,2*pi)
estimated_phaselist_l=mod.(estimated_phaselist_l,2*pi)

cosinor_grng=Compile_MultiCore_Cosinor_Statistics(fullnonseed_data_grng,estimated_phaselist_g,4,24)
cosinor_laval=Compile_MultiCore_Cosinor_Statistics(fullnonseed_data_laval,estimated_phaselist_l,4,24)

sig_grng=Filter_Cosinor_Output(cosinor_grng,.05,0,1.66)
sig_laval=Filter_Cosinor_Output(cosinor_laval,.05,0,1.66)

common_g_l=intersect(sig_grng[2:end,1],sig_laval[2:end,1])
rows_laval_g_l=findin(sig_laval[:,1],common_g_l)
rows_grng_g_l=findin(sig_grng[:,1],common_g_l)

use_grng_g_l=mod.(sig_grng[rows_grng_g_l,6],2*pi)
use_laval_g_l=mod.(sig_laval[rows_laval_g_l,6],2*pi)


use_laval_adj1,estimated_phaselist_l_adj1=best_shift_cos2(use_laval_g_l,use_grng_g_l,estimated_phaselist_l,"radians")

scatter(use_laval_adj1,use_grng_g_l,s=.5,color="DarkBlue")

cosinor_laval=Compile_MultiCore_Cosinor_Statistics(fullnonseed_data_laval,estimated_phaselist_l_adj1,4,24)
####################################
# Align with Ebox (Par-BZip) phase
#####################################
eboxgenes=["DBP","HLF","TEF"]

eboxphases_laval=cosinor_laval[findin(cosinor_laval[:,2],eboxgenes),:]
eboxphases_grng=cosinor_grng[findin(cosinor_grng[:,2],eboxgenes),:]

laval_criteria=((eboxphases_laval[:,4].<.05) & ( eboxphases_laval[:,11].>1.25) & ( eboxphases_laval[:,9].>100))
grng_criteria=((eboxphases_grng[:,4].<.05) & ( eboxphases_grng[:,11].>1.25) & ( eboxphases_laval[:,9].>100))

eboxphases_laval=eboxphases_laval[findin(laval_criteria,true),:]
eboxphases_grng=eboxphases_grng[findin(grng_criteria,true),:]

eboxphases_laval=Array{Float64}(eboxphases_laval[:,6])
eboxphases_grng=Array{Float64}(eboxphases_grng[:,6])

eboxphases=append!(eboxphases_laval,eboxphases_grng)
eboxphase=Circular_Mean(eboxphases)

estimated_phaselist_g_adj_final=mod.(estimated_phaselist_g .- eboxphase+(pi),2*pi)  ##ebox genes in mouse lung peak on average CT 11.5 
estimated_phaselist_l_adj_final=mod.(estimated_phaselist_l_adj1 .- eboxphase+(pi),2*pi)  ##ebox genes in mouse lung peak on average CT 11.5 



Nday=length(findin((estimated_phaselist_l_adj_final .>pi),true))
Nnight=length(findin((estimated_phaselist_l_adj_final .<pi),true))

if (Nday<Nnight)
    estimated_phaselist_l_adj_final=mod(2*pi-estimated_phaselist_l_adj_final,2*pi)
    estimated_phaselist_g_adj_final=mod(2*pi-estimated_phaselist_g_adj_final,2*pi)
end



###################################
cosinor_grng=Compile_MultiCore_Cosinor_Statistics(fullnonseed_data_grng,estimated_phaselist_g_adj_final,4,24)
cosinor_laval=Compile_MultiCore_Cosinor_Statistics(fullnonseed_data_laval,estimated_phaselist_l_adj_final,4,24)

nsig_grng=Filter_Cosinor_Output(cosinor_grng,.05,0,1.66)
nsig_laval=Filter_Cosinor_Output(cosinor_laval,.05,0,1.66)

common_g_l=intersect(nsig_grng[2:end,1],nsig_laval[2:end,1])

cd(outdir);
#writecsv("Laval_Sample_Phaselist.csv",estimated_phaselist_l_adj_final);
#writecsv("GRNG_Sample_Phaselist.csv",estimated_phaselist_g_adj_final);
#writecsv("Laval_Cosinor_Output.csv",cosinor_laval);
#writecsv("GRNG_Cosinor_Output.csv",cosinor_grng);

############################################################
sig_laval=Filter_Cosinor_Output(cosinor_laval,.05,0,1.666)
sig_grng=Filter_Cosinor_Output(cosinor_grng,.05,0,1.666)

############################################################
common_g_l=intersect(sig_grng[2:end,1],sig_laval[2:end,1])

rows_laval_g_l=findin(sig_laval[:,1],common_g_l)
rows_grng_g_l=findin(sig_grng[:,1],common_g_l)

use_grng_g_l=sig_grng[rows_grng_g_l,6]
use_laval_g_l=sig_laval[rows_laval_g_l,6]
############################################################
# Acrophase Comparison
############################################################
close()
ylabp=[0,pi/2,pi,3*pi/2,2*pi]
ylabs=[0, "","π","","2π"]
xlabp=[0,pi/2,pi,3*pi/2,2*pi]
xlabs=[0, "","π","","2π"]

scatter(mod(use_grng_g_l,2*pi),mod(use_laval_g_l,2*pi),s=.5,color="DarkBlue")
xticks(xlabp, xlabs, fontsize=18)
yticks(ylabp, ylabs, fontsize=18)
#xlabel("GRNG Acrophase (radians)", fontsize=18)
#ylabel("Laval Acrophase (radians)", fontsize=18)


#scatter(use_grng_g_l,use_laval_g_l,s=.5,color="DarkBlue")

############################################################
# Sample Historgrams
############################################################
close()
subplot(1,2,1)
plt[:hist](estimated_phaselist_g_adj_final,normed=1,alpha=.5,color="DarkGreen")
xlabel("Phase of Sample Collection (GRNG)", fontsize=18)
ylabel("Probability", fontsize=18)
subplot(1,2,2)
plt[:hist](estimated_phaselist_l_adj_final,normed=1,alpha=.5,color="DarkBlue")
ylabel("Probability", fontsize=18)
xlabel("Phase of Sample Collection (Laval)", fontsize=18)
############################################################
close()
plt[:hist](estimated_phaselist_l_adj_final,normed=1,alpha=.5,color="DarkBlue")
xticks(xlabp, xlabs)
ylabel("Probability", fontsize=18)
xlabel("Phase of Sample Collection (Laval)", fontsize=18)
############################################################
# Sample Historgrams - Laval - No annotations
############################################################
close()
plt[:hist](estimated_phaselist_l_adj_final,normed=1,alpha=.5,color="DarkBlue")
xticks(xlabp, xlabs)
############################################################
# Sample Historgrams - All- No annotations
############################################################
close()
plt[:hist](vcat(estimated_phaselist_l_adj_final,estimated_phaselist_g_adj_final),normed=1,alpha=.5,color="DarkBlue")
xticks(xlabp, xlabs)
ylabel("Probability", fontsize=18)
xlabel("Phase of Sample Collection (Laval & GRNG)", fontsize=18)
###################################
r_grng=Filter_Cosinor_Output(cosinor_grng,.05,.65,1.66)
r_laval=Filter_Cosinor_Output(cosinor_laval,.05,.65,1.66)

common_g_l_r=intersect(r_grng[2:end,2],r_laval[2:end,2])
common_g_l_r[common_g_l_r .!=""]

common_g_l_r=intersect(r_grng[2:end,1],r_laval[2:end,1])


clockgenes=["CLOCK","CRY2","CRY1","DBP","TEF","HLF","EFNB2","ADAM9","E4F1","ACE","NR1D1","NR1D2","RORA"]
aclockrows=findin(alldata_symbols_g,clockgenes);
clockrows=aclockrows
clockrows=aclockrows[[23,4,27,21,6,16,8,20,17]]
clockrows=aclockrows[[23,4,27,8,20,17]]
clockdata_g=alldata_data_g[clockrows,:];
clockdata_l=alldata_data_l[clockrows,:];

cosinor_clock_g=cosinor_grng[clockrows+1,:];  ## the +1 is because the first row of this array is the headings
cosinor_clock_l=cosinor_laval[clockrows+1,:];  ## the +1 is because the first row of this array is the headings

clockannotations=alldata_symbols_g[clockrows,:];

clockdata_g=alldata_data_g[clockrows,:];
clockdata_l=alldata_data_l[clockrows,:];

cosinor_clock_g=cosinor_grng[clockrows+1,:];  ## the +1 is because the first row of this array is the headings
cosinor_clock_l=cosinor_laval[clockrows+1,:];  ## the +1 is because the first row of this array is the headings

clockannotations=alldata_symbols_g[clockrows,:];
close()

close()

xlabp=[0,pi/2,pi,3*pi/2,2*pi]
xlabs=[0, "","π","","2π"]

for n in 1:6
          m1=2*n-1
          m2=2*n
          subplot(2,6,m1)

          ymean=mean(clockdata_g[n,:])
          ymax=minimum([ymean+3*std(clockdata_g[n,:]),maximum(clockdata_g[n,:])])
          ymin=maximum([0,ymean-3*std(clockdata_g[n,:])])
          
          ymaxstar=Integer(100*div(ymax,100))
          yminstar=Integer(100*div(ymin,100))
          ymedstar=Integer((yminstar+ymaxstar)/2)
          
          ylabp=[yminstar,ymedstar,ymaxstar]
          ylabs=[yminstar,ymedstar,ymaxstar]

          axis([0,2*pi,ymin,ymax])
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs, fontsize=10)


          scatter(estimated_phaselist_g_adj_final,clockdata_g[n,:],alpha=.75,s=7,color="DarkBlue")

          title(clockannotations[n], fontsize=18)
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs, fontsize=10)

          PrbPhase,PrbAmp,PrbMean=cosinor_clock_g[n,[6,7,8]]
          sest=linspace(0,2*pi,100)
          synth=PrbAmp*cos.(sest-PrbPhase)+PrbMean
          plot(sest,synth,"r-",lw=2)
        

          subplot(2,6,m2)

          ymean=0
          ymax=0
          ymin=0

          ymean=mean(clockdata_l[n,:])
          ymax=minimum([ymean+3*std(clockdata_l[n,:]),maximum(clockdata_l[n,:])])
          ymin=maximum([0,ymean-3*std(clockdata_l[n,:])])
          println(yminstar,"   ",ymedstar,"    ",ymaxstar)

          ymaxstar=Integer(100*div(ymax,100))
          yminstar=Integer(100*div(ymin,100))
          ymedstar=Integer((yminstar+ymaxstar)/2)
          
          ylabp=[yminstar,ymedstar,ymaxstar]
          ylabs=[yminstar,ymedstar,ymaxstar]

          axis([0,2*pi,ymin,ymax])
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs, fontsize=10)

          scatter(estimated_phaselist_l_adj_final,clockdata_l[n,:],alpha=.75,s=7,color="DarkGreen")

          title(clockannotations[n], fontsize=18)


          PrbPhase,PrbAmp,PrbMean=cosinor_clock_l[n,[6,7,8]]
          sest=linspace(0,2*pi,100)
          synth=PrbAmp*cos.(sest-PrbPhase)+PrbMean
          plot(sest,synth,"r-",lw=2)

end


#suptitle("",fontsize=22)



clockgenes=["KDR","ACE","MAP4K3","SMAD7","AQP4"]
clockrows=findin(alldata_symbols_g,clockgenes);
clockrows=clockrows[[1,4,5,7,11]]
clockdata_g=alldata_data_g[clockrows,:];
clockdata_l=alldata_data_l[clockrows,:];

cosinor_clock_g=cosinor_grng[clockrows+1,:];  ## the +1 is because the first row of this array is the headings
cosinor_clock_l=cosinor_laval[clockrows+1,:];  ## the +1 is because the first row of this array is the headings

clockannotations=alldata_symbols_g[clockrows,:];
close()
#############################

for n in 1:5
          m1=2*n-1
          m2=2*n
          subplot(5,2,m1)

          ymean=mean(clockdata_g[n,:])
          ymax=minimum([ymean+3*std(clockdata_g[n,:]),maximum(clockdata_g[n,:])])
          ymin=maximum([0,ymean-3*std(clockdata_g[n,:])])
          
          ymaxstar=Integer(100*div(ymax,100))
          yminstar=Integer(100*div(ymin,100))
          ymedstar=Integer((yminstar+ymaxstar)/2)
          
          ylabp=[yminstar,ymedstar,ymaxstar]
          ylabs=[yminstar,ymedstar,ymaxstar]

          axis([0,2*pi,ymin,ymax])
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs, fontsize=10)


          scatter(estimated_phaselist_g_adj_final,clockdata_g[n,:],alpha=.75,s=7,color="DarkBlue")

          title(clockannotations[n], fontsize=18)
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs, fontsize=10)

          PrbPhase,PrbAmp,PrbMean=cosinor_clock_g[n,[6,7,8]]
          sest=linspace(0,2*pi,100)
          synth=PrbAmp*cos.(sest-PrbPhase)+PrbMean
          plot(sest,synth,"r-",lw=2)
        

          subplot(5,2,m2)

          ymean=0
          ymax=0
          ymin=0

          ymean=mean(clockdata_l[n,:])
          ymax=minimum([ymean+3*std(clockdata_l[n,:]),maximum(clockdata_l[n,:])])
          ymin=maximum([0,ymean-3*std(clockdata_l[n,:])])
          println(yminstar,"   ",ymedstar,"    ",ymaxstar)

          ymaxstar=Integer(100*div(ymax,100))
          yminstar=Integer(100*div(ymin,100))
          ymedstar=Integer((yminstar+ymaxstar)/2)
          
          ylabp=[yminstar,ymedstar,ymaxstar]
          ylabs=[yminstar,ymedstar,ymaxstar]

          axis([0,2*pi,ymin,ymax])
          xticks(xlabp, xlabs)
          yticks(ylabp, ylabs, fontsize=10)

          scatter(estimated_phaselist_l_adj_final,clockdata_l[n,:],alpha=.75,s=7,color="DarkGreen")

          title(clockannotations[n], fontsize=18)


          PrbPhase,PrbAmp,PrbMean=cosinor_clock_l[n,[6,7,8]]
          sest=linspace(0,2*pi,100)
          synth=PrbAmp*cos.(sest-PrbPhase)+PrbMean
          plot(sest,synth,"r-",lw=2)

end
########################################
###########################
# Demonstration that allowing more itterations of neural network does not much change fit
#########################
function refine_order(l_outs::Integer,l_norm_seed_data::Array{Float64,2},NET,epochs=50000)
    scalefactor=1;
    besterror=10E20
    bestnet=NET;

    Train_Momentum_Stochastic!(l_norm_seed_data,NET,10,.1,.5,0.0005,epochs)
    global_circ_sse=Train_Bold!(l_norm_seed_data,NET,.1,0.0005,epochs)  
    besterror=global_circ_sse   
    print(global_circ_sse)

    total_sse,onedlinear_sse=Default_Metrics(l_norm_seed_data,l_outs)
    besterror=besterror*2
    global_metrics=[besterror, besterror/(total_sse-besterror), besterror/(onedlinear_sse-besterror)]

    estimated_phaselist=ExtractPhase(l_norm_seed_data,bestnet);
    estimated_phaselist=mod.(estimated_phaselist + 2*pi,2*pi)
    estimated_phaselist,bestnet,global_metrics
end 
########################
estimated_phaselist_gh,bestnet_gh,global_var_metrics_gh=refine_order(outs_g,norm_seed_data_g,bestnet_g,30000); 
estimated_phaselist_lh,bestnet_lh,global_var_metrics_lh=refine_order(outs_l,norm_seed_data_l,bestnet_l,30000); 

close()
subplot(1,2,1)
scatter(estimated_phaselist_g,estimated_phaselist_gh)
subplot(1,2,2)
scatter(estimated_phaselist_l,estimated_phaselist_lh)

end
