module CYCLOPS_2a_Seed
export clean_data!
export getseed
export dispersion
export dispersion!
export getseed_mca
export setdiff
export getseed_homologuesymbol
export getseed_homologueprobe
export getseed_homologuesymbol_brain

function clean_data!(data::Array{Float64,2},bluntpercent=0.99)
	ngenes,nsamples=size(data)
	nfloor=1+floor((1-bluntpercent)*nsamples)
	nceiling=ceil(bluntpercent*nsamples)
	for row=1:ngenes
		sorted=sort(vec(data[row,:]))
		vfloor=sorted[nfloor]
		vceil=sorted[nceiling]
		for sample =1:nsamples
			data[row,sample]=max(vfloor,data[row,sample])
			data[row,sample]=min(vceil ,data[row,sample])
		end
	end
	data
end

#############################################

function getseed(data::Array{Any,2},symbol_list,maxcv=.75,mincv=.07,minmean=500,blunt=.99)
	data_symbols=data[2:end ,2]
	data_data=data[2:end , 4:end];
	data_data=float64(data_data)
	data_data=clean_data!(data_data,blunt)
	ngenes,namples=size(data_data)

	gene_means=mean(data_data,2)
	gene_sds=std(data_data,2)
	gene_cvs= gene_sds ./ gene_means

	criteria1=findin(data_symbols,symbol_list)
	criteria2=findin((gene_means .> minmean),true)
	criteria3=findin((gene_cvs .> mincv),true)
	criteria4=findin((gene_cvs .< maxcv),true)

	allcriteria=intersect(criteria1,criteria2,criteria3,criteria4)
	seed_data=data_data[allcriteria,:]
	seed_symbols=data_symbols[allcriteria,:]
	seed_symbols, seed_data
end

##############################################


function getseed_mca(data::Array{Any,2},probe_list,maxcv=.75,mincv=.07,minmean=500,blunt=.99)
	
	data_symbols=data[2:end ,2];
	data_probes=data[2:end ,1];

	data_data=data[2:end , 4:end];
	data_data=float64(data_data)
	data_data=clean_data!(data_data,blunt)
	ngenes,namples=size(data_data)

	gene_means=mean(data_data,2)
	gene_sds=std(data_data,2)
	gene_cvs= gene_sds ./ gene_means

	criteria1=findin(data_probes,probe_list)
	criteria2=findin((gene_means .> minmean),true)
	criteria3=findin((gene_cvs .> mincv),true)
	criteria4=findin((gene_cvs .< maxcv),true)

	allcriteria=intersect(criteria1,criteria2,criteria3,criteria4)
	seed_data=data_data[allcriteria,:]
	seed_symbols=data_symbols[allcriteria,:]
	seed_symbols, seed_data
end
###############


function getseed_homologuesymbol(data::Array{Any,2},symbol_list,maxcv=.75,mincv=.07,minmean=500,blunt=.99)
	
	data_symbols=data[2:end ,2];
	data_probes=data[2:end ,1];

	data_data=data[2:end , 3:end];
	data_data=float64(data_data)
	data_data=clean_data!(data_data,blunt)
	ngenes,namples=size(data_data)

	gene_means=mean(data_data,2)
	gene_sds=std(data_data,2)
	gene_cvs= gene_sds ./ gene_means

	criteria1=findin(data_symbols,symbol_list)
	criteria2=findin((gene_means .> minmean),true)
	criteria3=findin((gene_cvs .> mincv),true)
	criteria4=findin((gene_cvs .< maxcv),true)

	allcriteria=intersect(criteria1,criteria2,criteria3,criteria4)
	seed_data=data_data[allcriteria,:]
	seed_symbols=data_symbols[allcriteria,:]
	seed_symbols, seed_data
end

##############################################


function getseed_homologueprobe(data::Array{Any,2},probe_list,maxcv=.75,mincv=.07,minmean=500,blunt=.99)
	
	data_symbols=data[2:end ,2];
	data_probes=data[2:end ,1];

	data_data=data[2:end , 3:end];
	data_data=float64(data_data)
	data_data=clean_data!(data_data,blunt)
	ngenes,namples=size(data_data)

	gene_means=mean(data_data,2)
	gene_sds=std(data_data,2)
	gene_cvs= gene_sds ./ gene_means

	criteria1=findin(data_probes,probe_list)
	criteria2=findin((gene_means .> minmean),true)
	criteria3=findin((gene_cvs .> mincv),true)
	criteria4=findin((gene_cvs .< maxcv),true)

	allcriteria=intersect(criteria1,criteria2,criteria3,criteria4)
	seed_data=data_data[allcriteria,:]
	seed_symbols=data_symbols[allcriteria,:]
	seed_symbols, seed_data
end

##############################################

function dispersion!(data::Array{Float64,2})
	ngenes,nsamples=size(data)
	for gene=1:ngenes
		genemean=mean(data[gene,:])
		for sample=1:nsamples
			data[gene,sample]=(data[gene,sample]-genemean)/genemean
		end
	end
	data
end

##############################################

function dispersion(data::Array{Float64,2})
	ngenes,nsamples=size(data)
	ndata=zeros(ngenes,nsamples)
	for gene=1:ngenes
		genemean=mean(data[gene,:])
		for sample=1:nsamples
			ndata[gene,sample]=(data[gene,sample]-genemean)/genemean
		end
	end
	ndata
end

##############################################
function setdiff(x,y)
       	z= [x[1]]
       	for element in x
       		if !in(element,y) 
       			push!(z,element)
       		end
       	end
    z=z[2:length(z)]
    z
end	
	
###############


function getseed_homologuesymbol_brain(data::Array{Any,2},symbol_list,maxcv=.75,mincv=.07,minmean=500,blunt=.99)
	
	data_symbols=data[4:end ,2];
	data_probes=data[4:end ,1];

	data_data=data[4:end , 4:end];
	data_data=float64(data_data)
	data_data=clean_data!(data_data,blunt)
	ngenes,namples=size(data_data)

	gene_means=mean(data_data,2)
	gene_sds=std(data_data,2)
	gene_cvs= gene_sds ./ gene_means

	criteria1=findin(data_symbols,symbol_list)
	criteria2=findin((gene_means .> minmean),true)
	criteria3=findin((gene_cvs .> mincv),true)
	criteria4=findin((gene_cvs .< maxcv),true)

	allcriteria=intersect(criteria1,criteria2,criteria3,criteria4)
	seed_data=data_data[allcriteria,:]
	seed_symbols=data_symbols[allcriteria,:]
	seed_symbols, seed_data
end

##############################################

end