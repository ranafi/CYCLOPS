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

using CYCLOPS_2a_AutoEncoderModule
using CYCLOPS_2a_PreNPostprocessModule



using CYCLOPS_2a_MCA
using CYCLOPS_2a_MultiCoreModule_Smooth
using CYCLOPS_2a_Seed
using CYCLOPS_2a_CircularStats_U

using PyPlot

indir	=			string(basedir,"/Documents/MouseCircadianAtlas/Data");
homologuedir	=	string(basedir,"/Documents/MouseCircadianAtlas/Data");
outdir	=			string(basedir,"/Google Drive/CYCLOPS_OUTPUT_FINAL_DISP_DISP/MCA")

############################################################
Frac_Var=0.85 # Set Number of Dimensions of SVD to maintain this fraction of variance
DFrac_Var=0.03 # Set Number of Dimensions of SVD to so that incremetal fraction of variance of var is at least this much
N_best =  20  # Number of random initial conditions to try for each optimization

total_background_num=40; # Number of background runs for global background refrence statistic
#total_background_num=200; # Number of background runs for global background refrence statistic
n_cores=5; # Number of machine cores

############################################################
cd(homologuedir);
seedinfo=readcsv("MCA_CyclingSeeds.csv");
seedinfo[2:end,15]=int(seedinfo[2:end,15])
#####################################################################################################
Seed_MinCV 				= 0.14
Seed_MaxCV 				= .75
MaxSeeds 				= 10000
Seed_Blunt				=.95
###########################################################


srand(1234);

cd(indir);
file_list=readcsv("DataFilesOrdered.csv");
n=0
estimated_phaselist1=0;
fullnonseed_data=0
file_name=0;

truetimes=linspace(18,64,24) #This is the annotated time of collection for each column"
plottimes=mod(truetimes,24) #Returning collection time modulo 24
plottimes[plottimes .== 0]=24  #Choosing to plot time 0 as 24 (for convenience)


for file_name in file_list
#file_name=file_list[4]
	print(file_name)

	tissue=split(split(file_name,"_")[2],".")[1]
	n=n+1
	fullnonseed_data	=	readcsv(file_name);
	print(file_name)
	alldata_probes		=	fullnonseed_data[2:end ,1];


	seedinfo_tissue_col			=   int(findin(seedinfo[1,1:end],[tissue]))[1];
	seedinfo_nontissue_cols		=	setdiff([3:13],[seedinfo_tissue_col]);


	tissue_thresholds 			=	hcat(seedinfo[2:end,1],sum((seedinfo[2:end,seedinfo_tissue_col] .<= 0.05),2))
	non_tissue_thresholds 		=	hcat(seedinfo[2:end,1],sum((seedinfo[2:end,seedinfo_nontissue_cols] .<= 0.05),2))

	
	alldata_data=float(fullnonseed_data[2:end,4:end])
	nprobes=size(alldata_probes)[1]
	cutrank=nprobes-MaxSeeds

	Seed_MinMean= (sort(vec(mean(alldata_data,2))))[cutrank]
	println(Seed_MinMean)

	seed_list1			=	alldata_probes
	seed_list2			=	tissue_thresholds[[false,tissue_thresholds[2:end,2] .== 1],1]
	seed_list3			=	non_tissue_thresholds[[false,non_tissue_thresholds[2:end,2] .>= 8],1]

	seed_data_dispersion1, seed_symbols1				=	CYCLOPS_MCA_DataPrepare(fullnonseed_data,seed_list1,Seed_MaxCV,Seed_MinCV,Seed_MinMean,Seed_Blunt)
	outs1, norm_seed_data1								=	GetEigenGenes(seed_data_dispersion1,Frac_Var,DFrac_Var,30)
	
	seed_data_dispersion2, seed_symbols2				=	CYCLOPS_MCA_DataPrepare(fullnonseed_data,seed_list2,Seed_MaxCV,Seed_MinCV,Seed_MinMean,Seed_Blunt)
	outs2, norm_seed_data2								=	GetEigenGenes(seed_data_dispersion2,Frac_Var,DFrac_Var,30)

	seed_data_dispersion3, seed_symbols3				=	CYCLOPS_MCA_DataPrepare(fullnonseed_data,seed_list3,Seed_MaxCV	,Seed_MinCV,Seed_MinMean,Seed_Blunt)
	outs3, norm_seed_data3								=	GetEigenGenes(seed_data_dispersion3,Frac_Var,DFrac_Var,30)


	a1=@spawn CYCLOPS_Order(outs1,norm_seed_data1,N_best);
	a2=@spawn CYCLOPS_Order(outs2,norm_seed_data2,N_best);
	a3=@spawn CYCLOPS_Order(outs3,norm_seed_data3,N_best);
	

	estimated_phaselist1,bestnet1,global_var_metrics1=fetch(a1);
	estimated_phaselist2,bestnet2,global_var_metrics2=fetch(a2);
	estimated_phaselist3,bestnet3,global_var_metrics3=fetch(a3);

	estimated_phaselist1								=	mod(estimated_phaselist1 + 2*pi,2*pi)
	estimated_phaselist2								=	mod(estimated_phaselist2 + 2*pi,2*pi)
	estimated_phaselist3								=	mod(estimated_phaselist3 + 2*pi,2*pi)

	global_smooth_metrics1								=	smoothness_measures(seed_data_dispersion1,norm_seed_data1,estimated_phaselist1)
	global_smooth_metrics2								=	smoothness_measures(seed_data_dispersion2,norm_seed_data2,estimated_phaselist2)
	global_smooth_metrics3								=	smoothness_measures(seed_data_dispersion3,norm_seed_data3,estimated_phaselist3)

	print("\r")

	println(tissue)
    println(global_smooth_metrics1)
    println(global_smooth_metrics2)
    println(global_smooth_metrics3)
    
	global_metrics1=[global_var_metrics1]
	global_metrics2=[global_var_metrics2]
	global_metrics3=[global_var_metrics3]

	pv1=[1,1,1]
	pv2=[1,1,1]
	pv3=[1,1,1]

#	if (global_smooth_metrics1[1]<1)
#	    pv1=multicore_backgroundstatistics_global_eigen(seed_data_dispersion1,outs1,N_best,total_background_num,global_metrics1)
#	end
	
#	if (global_smooth_metrics2[1]<1)
#	    pv2=multicore_backgroundstatistics_global_eigen(seed_data_dispersion2,outs2,N_best,total_background_num,global_metrics2)
#	end
	    
#   if (global_smooth_metrics3[1]<1)
#	    pv3=multicore_backgroundstatistics_global_eigen(seed_data_dispersion3,outs3,N_best,total_background_num,global_metrics3)
#	end

	println(pv1[3])
	println(pv2[3])
	println(pv3[3])

 	shiftephaselist1=best_shift_cos(estimated_phaselist1,truetimes,"hours")
	shiftephaselist2=best_shift_cos(estimated_phaselist2,truetimes,"hours")
	shiftephaselist3=best_shift_cos(estimated_phaselist3,truetimes,"hours")

	subplot(4,3,n)

   	correlations1=Jammalamadka_Circular_CorrelationMeasures(2*pi*truetimes/24,float(mod(shiftephaselist1,2*pi)))
    correlations2=Jammalamadka_Circular_CorrelationMeasures(2*pi*truetimes/24,float(mod(shiftephaselist2,2*pi)))
    correlations3=Jammalamadka_Circular_CorrelationMeasures(2*pi*truetimes/24,float(mod(shiftephaselist3,2*pi)))
    

    scatter(plottimes,shiftephaselist1,color="DarkRed",alpha=.75,s=14)
 	scatter(plottimes,shiftephaselist2,color="DarkGreen",alpha=.75,s=14)
 	scatter(plottimes,shiftephaselist3,color="DarkBlue",alpha=.75,s=14)

 	ylabp=[0,pi/2,pi,3*pi/2,2*pi]
    ylabs=[0, "","π","","2π"]
    xlabp=[0,6,12,18,24]
    xlabs=["0", "6","12","18","24"]
 		
 	xticks(xlabp, xlabs)
 	yticks(ylabp, ylabs)
	title(tissue)

	circ
	if (pv1[3]<=.05)
		text(26,4.14,"*",color="DarkRed",fontsize=20)
	end
	text(25,5.14,string(round(correlations1[2],2)),color="DarkRed",fontsize=11)

	if (pv2[3]<=.05)
		text(26,2.14,"*",color="DarkGreen",fontsize=20)
	end
	text(25,3.14,string(round(correlations2[2],2)),color="DarkGreen",fontsize=11)

	if (pv3[3]<=.05)
		text(26,0.14,"*",color="DarkBlue",fontsize=20)
	end	
    text(25,1.14,string(round(correlations3[2],2)),color="DarkBlue",fontsize=11)
end
cd(outdir);

suptitle("CYCLOPS Ordering of the Mouse Circadian Atlas",fontsize=22)

text(32,4.0*pi,	"All Probes",fontsize=17,color="DarkRed")
text(32,3.75*pi,"Cycles in this tissue",fontsize=17,color="DarkGreen")
text(32,3.5*pi,	"Cycles in 75% of others",fontsize=17,color="DarkBlue")
text(-75,7*pi,"CYCLOPS predicted phase (radians)",rotation="vertical",fontsize=17)
text(-36,-2.9, "Time of Sample Collection (CT)",fontsize=17)

#cd(outdir);
#savefig("MCA_ReconstructionsUncheck_2a_both.pdf")
#################################
#################################
