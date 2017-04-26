module	CYCLOPS_2a_MCA

export  CYCLOPS_MCA_DataPrepare

using	CYCLOPS_2a_Seed
using	CYCLOPS_2a_PreNPostprocessModule
#############################################################################
#############################################################################
function CYCLOPS_MCA_DataPrepare(l_data::Array{Any,2},l_seed_list,maxcv=.75,mincv=.07,minmean=400,blunt=.95,Frac_Var=0.85, DFrac_Var=0.03)
	seed_symbols, seed_data = getseed_mca(l_data,l_seed_list,maxcv,mincv,minmean,blunt)
	seed_data=dispersion!(seed_data) #choose this option for 0 mean, %dispersion normalization
#	(outs,norm_seed_data)= PCA_Transform_Seed_Data(seed_data,Frac_Var,25) #choose this option for PCA dimension reduction (Eigenvectors with higher variance count more)
#	(outs,norm_seed_data)= GetEigenGenes(seed_data,Frac_Var,DFrac_Var,30); #choose this option for EIGEN dimension reduction (Eigenvectors count equally)
	seed_data, seed_symbols
#   outs,norm_seed_data
end 
#############################################################################
#############################################################################
end
