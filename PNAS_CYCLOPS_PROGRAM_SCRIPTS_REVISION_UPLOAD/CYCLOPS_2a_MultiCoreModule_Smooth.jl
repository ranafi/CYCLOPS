module CYCLOPS_2a_MultiCoreModule_Smooth

using StatsBase
using MultivariateStats
using CYCLOPS_2a_AutoEncoderModule
using CYCLOPS_2a_PreNPostprocessModule

export backgroundmetrics_global_eigen
export multicore_backgroundmetrics_global_eigen
export multicore_backgroundstatistics_global_eigen
export smoothness_measures, circ_diff, circ_diff_phases

###########################################################################
#  Note this is written to take adventage of a 6 core device
###########################################################################

function circ_diff(data::Array{Float64,2})
       	circd=hcat(diff(data,2),data[:,1]-data[:,end])
       	circd
end

###########################################################################
function circ_diff_phases(data::Array{Float64,1})
       	circd=[diff(data),data[1]+ (2*pi-data[end])]
       	circd
end

###########################################################################
###########################################################################
###########################################################################
function circ_diff_phases2(data)
	ans=zeros(length(data))


	ans[1]=((2*pi+data[2])-data[end])/2

		for i in 2:length(data)-1
			ans[i]=(data[i+1]-data[i-1])/2
		end

	ans[end]=((2*pi+data[1])-data[end-1])/2
	
	ans	
end


###########################################################################
function smoothness_measures(l_seeddata::Array{Float64,2},l_eigendata::Array{Float64,2},estimated_phaselist)
	estimated_phaselist1s=mod(estimated_phaselist,2*pi)
	use_order_c=sortperm(estimated_phaselist1s)
	use_order_l=sortperm(vec(l_eigendata[1,:]))


	new_phaselist=vec(estimated_phaselist1s[use_order_c])
	circ_seeddata=l_seeddata[:,use_order_c]
	circ_eigendata=l_eigendata[:,use_order_c]

	lin_seeddata=l_seeddata[:,use_order_l]
	lin_eigendata=l_eigendata[:,use_order_l]
	
	nsamp=size(l_seeddata)[2]

	gdiffs				=circ_diff(circ_seeddata)
	gdiffs2				=gdiffs .* gdiffs
	gdiffsm				=sqrt(sum(gdiffs2,1))
	num					=sum(gdiffsm)/nsamp

	gdiffs				=diff(lin_seeddata,2)
	gdiffs2				=gdiffs .* gdiffs
	gdiffsm				=sqrt(sum(gdiffs2,1))
	denom				=sum(gdiffsm)/(nsamp-1)
	measure_raw			=num/denom
	
	######################################################
	gdiffs				=circ_diff(circ_seeddata)
	gdiffs2				=gdiffs .* gdiffs
	gdiffsm				=(sum(gdiffs2,1))
	num					=sum(gdiffsm)/nsamp

	gdiffs				=diff(lin_seeddata,2)
	gdiffs2				=gdiffs .* gdiffs
	gdiffsm				=(sum(gdiffs2,1))
	denom				=sum(gdiffsm)/(nsamp-1)
	measure_raw_sq		=num/denom
	######################################################
	gdiffs				=circ_diff(circ_eigendata)
	gdiffs2				=gdiffs .* gdiffs
	gdiffsm				=sqrt(sum(gdiffs2,1))
	num					=sum(gdiffsm)/nsamp

	gdiffs				=diff(lin_eigendata,2)
	gdiffs2				=gdiffs .* gdiffs
	gdiffsm				=sqrt(sum(gdiffs2,1))
	denom				=sum(gdiffsm)/(nsamp-1)
	measure_eigen		=num/denom
	
	######################################################
	gdiffs				=circ_diff(circ_eigendata)
	gdiffs2				=gdiffs .* gdiffs
	gdiffsm				=(sum(gdiffs2,1))
	num					=sum(gdiffsm)/nsamp

	gdiffs				=diff(lin_eigendata,2)
	gdiffs2				=gdiffs .* gdiffs
	gdiffsm				=(sum(gdiffs2,1))
	denom				=sum(gdiffsm)/(nsamp-1)
	measure_eigen_sq	=num/denom
    ###########################################################################

	[measure_eigen,measure_eigen_sq,measure_raw,measure_raw_sq]

end


###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
#########################################################################

function backgroundmetrics_global_eigen(seed_ldata::Array{Float64,2},ESize::Integer,N_best::Integer,N_runs::Integer)

	background_global_metrics=zeros(N_runs,3)
	netsize=ESize
	
	for count=1:N_runs
		best_error=10E40;

		shuffled_seed_data=Row_Shuffle(seed_ldata)
		EigenShuffledData=GetNEigenGenes(shuffled_seed_data,ESize)
		estimated_phaselist,bestnet,variance_global_metrics=CYCLOPS_Order(ESize,EigenShuffledData,N_best);
		background_global_metrics[count,:]=variance_global_metrics
	end
	background_global_metrics
end

###########################################################################
function multicore_backgroundmetrics_global_eigen(seed_ldata::Array{Float64,2},ESize::Integer,N_best::Integer,N_trials::Integer)

	a1=@spawn backgroundmetrics_global_eigen(seed_ldata,ESize,N_best,int(N_trials/5));
	a2=@spawn backgroundmetrics_global_eigen(seed_ldata,ESize,N_best,int(N_trials/5));
	a3=@spawn backgroundmetrics_global_eigen(seed_ldata,ESize,N_best,int(N_trials/5));
	a4=@spawn backgroundmetrics_global_eigen(seed_ldata,ESize,N_best,int(N_trials/5));
	a5=@spawn backgroundmetrics_global_eigen(seed_ldata,ESize,N_best,int(N_trials/5));


	global1=fetch(a1);
	global2=fetch(a2);
	global3=fetch(a3);
	global4=fetch(a4);
	global5=fetch(a5);


	global_back=vcat(global1,global2,global3,global4,global5);
	global_back
end

###########################################################################
function multicore_backgroundstatistics_global_eigen(seed_ldata::Array{Float64,2},ESize::Integer,N_best::Integer,N_trials::Integer,testmetrics)
	
	backgroundmetrics=multicore_backgroundmetrics_global_eigen(seed_ldata,ESize,N_best,N_trials)

	gm1=backgroundmetrics[:,1]
	gm2=backgroundmetrics[:,2]
	gm3=backgroundmetrics[:,3]
	

	nless1=length(gm1[gm1 .<= testmetrics[1]]);
	nless2=length(gm2[gm2 .<= testmetrics[2]]);
	nless3=length(gm3[gm3 .<= testmetrics[3]]);


	p1=nless1/N_trials
	p2=nless2/N_trials
	p3=nless3/N_trials
	
	[p1,p2,p3]
	###Note p3 is the statistical measure to be used########
end

###########################################################################
end