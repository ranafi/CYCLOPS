module Extra_HCC_Functions


export getseed_woutputprobe
export getseed_winputprobe
export GetEigenGenes2
export Tumor_Compare
export Tumor_Compare2

using CYCLOPS_2a_Seed
using MultivariateStats
using Distributions

###################################################

#function clean_data!(data::Array{Float64,2},bluntpercent=0.99)
#    ngenes,nsamples=size(data)
#    nfloor=1+floor((1-bluntpercent)*nsamples)
#    nceiling=ceil(bluntpercent*nsamples)
#    for row=1:ngenes
#        sorted=sort(vec(data[row,:]))
#        vfloor=sorted[nfloor]
#       vceil=sorted[nceiling]
#        for sample =1:nsamples
#            data[row,sample]=max(vfloor,data[row,sample])
#            data[row,sample]=min(vceil ,data[row,sample])
#        end
#    end
#    data
#end

###################################################

function getseed_woutputprobe(data::Array{Any,2},symbol_list,maxcv=.75,mincv=.07,minmean=500,blunt=.99)
	data_symbols=data[2:end ,2]
	data_probes=data[2:end,1]
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
	seed_probes=data_probes[allcriteria,:]
	seed_symbols=data_symbols[allcriteria,:]
	seed_probes, seed_symbols, seed_data
end

###################################################
###################################################
function getseed_winputprobe(data::Array{Any,2},probe_list,blunt=.99)
	data_symbols=data[2:end ,2]
	data_probes=data[2:end,1]
	data_data=data[2:end , 4:end];
	data_data=float64(data_data)
	data_data=clean_data!(data_data,blunt)
	ngenes,namples=size(data_data)

	criteria1=findin(data_probes,probe_list)

	allcriteria=criteria1
	seed_data=data_data[allcriteria,:]
	seed_probes=data_probes[allcriteria,:]
	seed_symbols=data_symbols[allcriteria,:]
	seed_probes,seed_symbols, seed_data
end

###################################################
function GetEigenGenes2(numeric_data1::Array{Float64,2},numeric_data2::Array{Float64,2},fraction_var=.85,maxeig=50)
    Mod=fit(PCA,numeric_data1;pratio=fraction_var,maxoutdim=maxeig)
    ReductionDim=outdim(Mod)
    Transform1=transform(Mod,numeric_data1)
    Transform2=transform(Mod,numeric_data2)

    nrows=size(Transform1)

    vars1=var(Transform1,2)
    vars2=var(Transform2,2)

    Eigen1=Transform1 ./ sqrt(vars1)
    Eigen2=Transform2 ./ sqrt(vars2)
    
    ReductionDim, Eigen1,Eigen2
end
############################################################
###################################################
#####################################################################################################################
function Tumor_Compare(nml_expression::Array{Float64,2},tumor_expression::Array{Float64,2},nml_PREDICTED_PHASELIST::Array{Float64,1},tumor_PREDICTED_PHASELIST::Array{Float64,1})
    
    nml_len=size(nml_PREDICTED_PHASELIST)[1]
    tumor_len=size(tumor_PREDICTED_PHASELIST)[1]
    all_len=nml_len+tumor_len

    all_expression=vcat(nml_expression,tumor_expression)
    PREDICTED_PHASELIST=vcat(nml_PREDICTED_PHASELIST,tumor_PREDICTED_PHASELIST)

    SIN_All       =     sin(PREDICTED_PHASELIST);
    COS_All       =     cos(PREDICTED_PHASELIST);

    b_ALL         =     ones(size(PREDICTED_PHASELIST));
    b_TUMOR       =     vcat(zeros(nml_len),ones(tumor_len));


    SIN_TUMOR     =     vcat(zeros(nml_len),sin(tumor_PREDICTED_PHASELIST));
    COS_TUMOR     =     vcat(zeros(nml_len),cos(tumor_PREDICTED_PHASELIST));


    XSAME   =hcat(SIN_All,COS_All,b_ALL,b_TUMOR);
    XDIFF   =hcat(SIN_All,COS_All,SIN_TUMOR,COS_TUMOR,b_ALL,b_TUMOR);

    mod_same          =llsq(XSAME,all_expression,bias=false);
    mod_diff      =llsq(XDIFF,all_expression,bias=false);
    
    predict_same      =  XSAME*mod_same;
    predict_diff    =  XDIFF*mod_diff;

    sse_same          =sum(abs2(all_expression-predict_same));
    sse_diff          =sum(abs2(all_expression-predict_diff));

    f_metric            =((sse_same-sse_diff)/2)/((sse_diff)/(all_len-6)); #p2=6  ,p1=4   ,n=all_len
    pval                = 1-cdf(FDist(2,all_len-6),f_metric) 

    
    ###################
    #Get Nml Phase,Amp,Avg
    ####################
    SIN_NML_PHASELIST   =sin(nml_PREDICTED_PHASELIST);
    COS_NML_PHASELIST   =cos(nml_PREDICTED_PHASELIST);
    NML_XCOSINOR        =hcat(SIN_NML_PHASELIST,COS_NML_PHASELIST);
    NML_mod_cosinor     =llsq(NML_XCOSINOR,nml_expression);

    m_NML_cosinor, b_NML_osinor        = NML_mod_cosinor[1:end-1], NML_mod_cosinor[end];
    
    NML_sinterm, NML_costerm  = NML_mod_cosinor[1], NML_mod_cosinor[2];
    NML_acrophase=atan2(NML_sinterm,NML_costerm);
    NML_amp=sqrt(NML_sinterm^2+NML_costerm^2);
    
    ###################
    #Get Tumor Phase,Amp,Avg
    ####################
    
    SIN_TUMOR_PHASELIST   =sin(tumor_PREDICTED_PHASELIST);
    COS_TUMOR_PHASELIST   =cos(tumor_PREDICTED_PHASELIST);
    TUMOR_XCOSINOR        =hcat(SIN_TUMOR_PHASELIST,COS_TUMOR_PHASELIST);
    TUMOR_mod_cosinor     =llsq(TUMOR_XCOSINOR,tumor_expression);

    TUMOR_sinterm, TUMOR_costerm  = TUMOR_mod_cosinor[1], TUMOR_mod_cosinor[2];
    TUMOR_acrophase=atan2(TUMOR_sinterm,TUMOR_costerm);
    TUMOR_amp=sqrt(TUMOR_sinterm^2+TUMOR_costerm^2);



    [pval, NML_amp, TUMOR_amp, NML_acrophase, TUMOR_acrophase]

end
############################################################
###################################################
#####################################################################################################################
function Tumor_Compare2(nml_expression::Array{Float64,2},tumor_expression::Array{Float64,2},nml_PREDICTED_PHASELIST::Array{Float64,1},tumor_PREDICTED_PHASELIST::Array{Float64,1})
    
    nml_len=size(nml_PREDICTED_PHASELIST)[1]
    tumor_len=size(tumor_PREDICTED_PHASELIST)[1]
    all_len=nml_len+tumor_len

    all_expression=vcat(nml_expression,tumor_expression)
    PREDICTED_PHASELIST=vcat(nml_PREDICTED_PHASELIST,tumor_PREDICTED_PHASELIST)

    SIN_All       =     sin(PREDICTED_PHASELIST);
    COS_All       =     cos(PREDICTED_PHASELIST);

    b_ALL         =     ones(size(PREDICTED_PHASELIST));
    b_TUMOR       =     vcat(zeros(nml_len),ones(tumor_len));


    SIN_TUMOR     =     vcat(zeros(nml_len),sin(tumor_PREDICTED_PHASELIST));
    COS_TUMOR     =     vcat(zeros(nml_len),cos(tumor_PREDICTED_PHASELIST));


    XSAME   =hcat(SIN_All,COS_All,b_ALL,b_TUMOR);
    XDIFF   =hcat(SIN_All,COS_All,SIN_TUMOR,COS_TUMOR,b_ALL,b_TUMOR);

    mod_same      =llsq(XSAME,all_expression,bias=false);
    mod_diff      =llsq(XDIFF,all_expression,bias=false);
    
    predict_same      =  XSAME*mod_same;
    predict_diff    =  XDIFF*mod_diff;

    sse_same          =sum(abs2(all_expression-predict_same));
    sse_diff          =sum(abs2(all_expression-predict_diff));

    f_metric            =((sse_same-sse_diff)/2)/((sse_diff)/(all_len-6)); #p2=6  ,p1=4   ,n=all_len
    pval                = 1-cdf(FDist(2,all_len-6),f_metric) 

    NML_sinterm = mod_diff[1]
    NML_costerm = mod_diff[2]
    TUMOR_sinterm=mod_diff[1]+mod_diff[3]
    TUMOR_costerm=mod_diff[2]+mod_diff[4]
    
    NML_acrophase=atan2(NML_sinterm,NML_costerm);
    NML_amp=sqrt(NML_sinterm^2+NML_costerm^2);
    
    TUMOR_acrophase=atan2(TUMOR_sinterm,TUMOR_costerm);
    TUMOR_amp=sqrt(TUMOR_sinterm^2+TUMOR_costerm^2);



    [pval, NML_amp, TUMOR_amp, NML_acrophase, TUMOR_acrophase]

end


end