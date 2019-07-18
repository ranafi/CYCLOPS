module CYCLOPS_v6_2a_AutoEncoderModule

using StatsBase
using MultivariateStats

export linr, linr_deriv, circ
export Input_Layer, Output_Layer, BottleNeck_Layer
export Create_InputLayer, Create_OutputLayer, Create_BottleNeckLayer
export Layer_Connections, Initialize_Layer_Connections
export NeuralNetwork, Create_Network
export Feed_Forward!, Find_Partner
export Delta,   Find_Delta!, Find_Gradients!, Find_Total_Gradients!, Find_Total_Error
export ExtractPhase, ExtractProjection,Train_Bold!,Train_Momentum_Stochastic!
export CYCLOPS_Order,CYCLOPS_Order2,CYCLOPS_Order_base,CYCLOPS_Order_controlled,CYCLOPS_Order_n
export ExtractPhase_Unbalanced
export Default_Metrics

export UnbalancedNeuralNetwork, Create_Network_Unbalanced
export Feed_Forward_Unbalanced!, Find_Gradients_Unbalanced!, Find_Total_Gradients_Unbalanced!
export Train_Momentum_Stochastic_Unbalanced!, Train_Bold_Unbalanced!, CYCLOPS_Order_n_Unbalanced
export ExtractMagnitude_Unbalanced, ExtractMagnitude

####################################################
# identity activation function
function linr(z::Float64, dummy=1.0)
    z
end

# identity derivative
function linr_deriv(z::Float64, dummy=1.0)
    1.
end
#####################################################
# circular activation function
function circ(z::Float64,zstar::Float64)
  z/(sqrt(z^2+zstar^2))
end
######################################################################
type Input_Layer
    a::Array{Float64,1}           # activation/input value
end

function Create_InputLayer(in_dim::Integer)
    l=Input_Layer(zeros(in_dim))
    l
end
######################################################################
type Output_Layer
    z::Array{Float64,1}           # pre activation value
    a::Array{Float64,1}           # activation value

    a_func::Array{Function,1}     # activation function of node j in layer
    a_deriv::Array{Function,1}     # activation funciton derivative
end

function Create_OutputLayer(out_dim::Integer,activ_func::Function,activ_deriv::Function)
    z=zeros(out_dim)
    a=zeros(out_dim)
    a_func=repmat([activ_func],out_dim)
    a_deriv=repmat([activ_deriv],out_dim)
    Output_Layer(z,a,a_func,a_deriv)
end
######################################################################
function Find_Partner(x::Integer) # find index of partnered node
    grp=div(x-1,2)
    elm=mod(x-1,2)
    pelm=(1-elm)
    partner=1+(grp)*2+pelm
    partner
end

type BottleNeck_Layer
    z::Array{Float64,1}           # pre activation value
    a::Array{Float64,1}           # post activation value

    a_func::Array{Function,1}     # activation function of node j in the network
    jstar::Array{Int8,1}          # index of partnered node - if ciruclar node in layer
 end

function Create_BottleNeckLayer(layer_size::Integer,n_circ::Integer)
    z=zeros(layer_size)
    a=zeros(layer_size)

#    a_func=[repmat([circ],n_circ),repmat([linr],(layer_size-n_circ))]
     a_func_1=Array{Any}(repmat([circ],n_circ))
     a_func_2=Array{Any}(repmat([linr],(layer_size-n_circ)))
     a_func=Array{Function}(append!(a_func_1,a_func_2))

    jstar=Array{Integer}(zeros(layer_size))
    for i=1:n_circ
            jstar[i]=Find_Partner(i)
    end
    for i=(n_circ+1):layer_size
            jstar[i]=i
    end
    BottleNeck_Layer(z,a,a_func,jstar)
end
######################################################################
type Layer_Connections
    w::Array{Float64,2}   # w[j,k] weight of connection from node k in previous layer to node j in this layer
    b::Array{Float64,1}    # b[j] bias of node j in present layer
end

function Initialize_Layer_Connections(layer_dim::Integer,in_dim::Integer)
    w=randn(layer_dim,in_dim)
    b=randn(layer_dim)/100.
    Layer_Connections(w,b)
end
######################################################################
type NeuralNetwork
    dim ::Integer
    nbottle::Integer
    ncirc::Integer
    ilayer::Input_Layer
    blayer::BottleNeck_Layer
    olayer::Output_Layer
    c2::Layer_Connections
    c3::Layer_Connections
end

function Create_Network(nsize::Integer,bottle_size::Integer,n_circ::Integer)
    ilayer=Create_InputLayer(nsize)
    blayer=Create_BottleNeckLayer(bottle_size,n_circ)
    olayer=Create_OutputLayer(nsize,linr,linr_deriv)
    c2=Initialize_Layer_Connections(bottle_size,nsize)
    c3=Initialize_Layer_Connections(nsize,bottle_size)
    NeuralNetwork(nsize,bottle_size,n_circ,ilayer,blayer,olayer,c2,c3)
end

######################################################################
type UnbalancedNeuralNetwork
    indim::Integer
    outdim::Integer
    nbottle::Integer
    ncirc::Integer
    ilayer::Input_Layer
    blayer::BottleNeck_Layer
    olayer::Output_Layer
    c2::Layer_Connections
    c3::Layer_Connections
end

function Create_Network_Unbalanced(insize::Integer,outsize::Integer,bottle_size::Integer,n_circ::Integer)
    ilayer=Create_InputLayer(insize)
    blayer=Create_BottleNeckLayer(bottle_size,n_circ)
    olayer=Create_OutputLayer(outsize,linr,linr_deriv)
    c2=Initialize_Layer_Connections(bottle_size,insize)
    c3=Initialize_Layer_Connections(outsize,bottle_size)
    UnbalancedNeuralNetwork(insize,outsize,bottle_size,n_circ,ilayer,blayer,olayer,c2,c3)
end
######################################################################
function Feed_Forward!(data::Array{Float64,1},NN::NeuralNetwork)
    NN.ilayer.a=data    #set input nodes=data

    NN.blayer.z=(NN.c2.w)*(NN.ilayer.a)+(NN.c2.b)   #process through weights and biases
                                                    #to find preactivation value next layer

    for j=1:NN.ncirc                    #process through circular bottleneck activation function
        jstar=Find_Partner(j)
        NN.blayer.a[j]=NN.blayer.a_func[j](NN.blayer.z[j],NN.blayer.z[jstar])
    end                                     

    for j=(NN.ncirc+1):NN.nbottle           #process through non-circular bottleneck activation function
        NN.blayer.a[j]=NN.blayer.a_func[j](NN.blayer.z[j])
    end    
    
    NN.olayer.z=(NN.c3.w)*(NN.blayer.a)+(NN.c3.b)
    


    for j=1:NN.dim
        NN.olayer.a[j]=NN.olayer.a_func[j](NN.olayer.z[j])
    end
end

######################################################################
######################################################################
function Feed_Forward_Unbalanced!(inputdata::Array{Float64,1},UNN::UnbalancedNeuralNetwork)
    UNN.ilayer.a=inputdata    #set input nodes=data

    UNN.blayer.z=(UNN.c2.w)*(UNN.ilayer.a)+(UNN.c2.b)   #process through weights and biases
                                                    #to find preactivation value next layer

    for j=1:UNN.ncirc                    #process through circular bottleneck activation function
        jstar=Find_Partner(j)
        UNN.blayer.a[j]=UNN.blayer.a_func[j](UNN.blayer.z[j],UNN.blayer.z[jstar])
    end                                     

    for j=(UNN.ncirc+1):UNN.nbottle           #process through non-circular bottleneck activation function
        UNN.blayer.a[j]=UNN.blayer.a_func[j](UNN.blayer.z[j])
    end    
    
    UNN.olayer.z=(UNN.c3.w)*(UNN.blayer.a)+(UNN.c3.b)
    


    for j=1:UNN.outdim
        UNN.olayer.a[j]=UNN.olayer.a_func[j](UNN.olayer.z[j])
    end
end

######################################################################
type Delta #following Michael Nielson text delta=dCost/dz
    blayer::Array{Float64,1}
    olayer::Array{Float64,1}
end
######################################################################
function Find_Gradients!(data::Array{Float64,1},NN::NeuralNetwork)
    Feed_Forward!(data,NN)

    c2grads=Layer_Connections(0* NN.c2.w, 0*NN.c2.b)
    c3grads=Layer_Connections(0* NN.c3.w, 0*NN.c3.b)

    del_o=(-data+NN.olayer.a)  #simple difference between layer and goal
    err  = (.5*sum((del_o).^2))
    for j=1:NN.dim             #this assumes cost function = 1/2*SSE = 1/2 L2 Norm
        del_o[j]=NN.olayer.a_deriv[j](NN.olayer.z[j])*del_o[j]
        c3grads.b[j]=(del_o[j])
        for k=1:NN.nbottle
            c3grads.w[j,k]=(del_o[j])*(NN.blayer.a[k])
        end
    end


    r=zeros(NN.ncirc)
    for j=1:NN.ncirc
        jstar=Find_Partner(j)
        r[j]=sqrt((NN.blayer.z[j])^2+(NN.blayer.z[jstar])^2)
    end

    del_b=zeros(NN.nbottle)

    for j=1:NN.ncirc
        jstar=Find_Partner(j)
        dsum=0
        for k=1:NN.dim
            d=del_o[k] * 1/(r[j]^3) * (NN.c3.w[k,j] * (NN.blayer.z[jstar])^2 - NN.c3.w[k,jstar] * (NN.blayer.z[j])*(NN.blayer.z[jstar]))
            dsum=dsum+d
        end

        del_b[j]=dsum
        c2grads.b[j]=del_b[j]

        for k=1:NN.dim
            c2grads.w[j,k]=(del_b[j])*(NN.ilayer.a[k])
        end

    end
    for j=(NN.ncirc+1):NN.nbottle
#        println("shouldn't evaluate if only circular bottle")
        dsum=0
        for k=1:NN.dim
            d=NN.c3.w[k,j]*del_o[k]
            dsum=dsum+d
        end

        del_b[j]=dsum
        c2grads.b[j]=del_b[j]

        for k=1:NN.dim
            c2grads.w[j,k]=(del_b[j])*(NN.ilayer.a[k])
        end

    end
    c2grads , c3grads, err
end
######################################################################
function Find_Gradients_Unbalanced!(inputdata::Array{Float64,1},outputdata::Array{Float64,1},UNN::UnbalancedNeuralNetwork)
    Feed_Forward_Unbalanced!(inputdata,UNN)

    c2grads=Layer_Connections(0* UNN.c2.w, 0*UNN.c2.b)
    c3grads=Layer_Connections(0* UNN.c3.w, 0*UNN.c3.b)

    del_o=(-outputdata+UNN.olayer.a)  #simple difference between layer and goal
    err  = (.5*sum((del_o).^2))
    
    for j=1:UNN.outdim             #this assumes cost function = 1/2*SSE = 1/2 L2 Norm
        del_o[j]=UNN.olayer.a_deriv[j](UNN.olayer.z[j])*del_o[j]
        c3grads.b[j]=(del_o[j])
        for k=1:UNN.nbottle
            c3grads.w[j,k]=(del_o[j])*(UNN.blayer.a[k])
        end
    end


    r=zeros(UNN.ncirc)
    for j=1:UNN.ncirc
        jstar=Find_Partner(j)
        r[j]=sqrt((UNN.blayer.z[j])^2+(UNN.blayer.z[jstar])^2)
    end

    del_b=zeros(UNN.nbottle)

    for j=1:UNN.ncirc
        jstar=Find_Partner(j)
        dsum=0
        for k=1:UNN.outdim
            d=del_o[k] * 1/(r[j]^3) * (UNN.c3.w[k,j] * (UNN.blayer.z[jstar])^2 - UNN.c3.w[k,jstar] * (UNN.blayer.z[j])*(UNN.blayer.z[jstar]))
            dsum=dsum+d
        end

        del_b[j]=dsum
        c2grads.b[j]=del_b[j]

        for k=1:UNN.indim
            c2grads.w[j,k]=(del_b[j])*(UNN.ilayer.a[k])
        end

    end
    for j=(UNN.ncirc+1):UNN.nbottle
#        println("shouldn't evaluate if only circular bottle")
        dsum=0
        for k=1:UNN.outdim
            d=UNN.c3.w[k,j]*del_o[k]
            dsum=dsum+d
        end

        del_b[j]=dsum
        c2grads.b[j]=del_b[j]

        for k=1:UNN.indim
            c2grads.w[j,k]=(del_b[j])*(UNN.ilayer.a[k])
        end

    end
    c2grads , c3grads, err
end

######################################################################
function Find_Total_Gradients!(data_matrix::Array{Float64,2},NN::NeuralNetwork)
    c2wt=zeros(size(NN.c2.w))
    c3wt=zeros(size(NN.c3.w))
    c2bt=zeros(size(NN.c2.b))
    c3bt=zeros(size(NN.c3.b))

    terr=0
    ntimepoints=size(data_matrix,2)
    for n=1:ntimepoints
        (c2gradients,c3gradients, err)=Find_Gradients!(data_matrix[:,n],NN)
       c2wt=c2wt+(c2gradients.w)/ntimepoints
       c3wt=c3wt+(c3gradients.w)/ntimepoints
       c2bt=c2bt+(c2gradients.b)/ntimepoints
       c3bt=c3bt+(c3gradients.b)/ntimepoints
       terr=terr+err/ntimepoints

    end
    c2tgrads=Layer_Connections(c2wt,c2bt)
    c3tgrads=Layer_Connections(c3wt,c3bt)

    c2tgrads,c3tgrads,terr
end


######################################################################
function Find_Total_Gradients_Unbalanced!(inputdata_matrix::Array{Float64,2},outputdata_matrix::Array{Float64,2},UNN::UnbalancedNeuralNetwork)
    c2wt=zeros(size(UNN.c2.w))
    c3wt=zeros(size(UNN.c3.w))
    c2bt=zeros(size(UNN.c2.b))
    c3bt=zeros(size(UNN.c3.b))

    terr=0
    ntimepoints=size(outputdata_matrix,2)
    ntimepoints2=size(inputdata_matrix,2)

    if (ntimepoints != ntimepoints2)
        println("error unequal number of input and output samples")
     end   
    for n=1:ntimepoints
        (c2gradients,c3gradients, err)=Find_Gradients_Unbalanced!(inputdata_matrix[:,n],outputdata_matrix[:,n],UNN)
        c2wt=c2wt+(c2gradients.w)/ntimepoints
        c3wt=c3wt+(c3gradients.w)/ntimepoints
        c2bt=c2bt+(c2gradients.b)/ntimepoints
        c3bt=c3bt+(c3gradients.b)/ntimepoints
        terr=terr+err/ntimepoints
    end

    c2tgrads=Layer_Connections(c2wt,c2bt)
    c3tgrads=Layer_Connections(c3wt,c3bt)

    c2tgrads,c3tgrads,terr
end


######################################################################
function Train_Bold!(data_matrix::Array{Float64,2},NN::NeuralNetwork,rate=.3,tol=0.0001,epochs=5000)
#   tic()

    (c2tgrads,c3tgrads,e0)=Find_Total_Gradients!(data_matrix,NN)
    max=1
    n=0
    e1=1.0
    if (NN.nbottle != NN.ncirc)
        rate=rate/NN.dim
    end
    
    while (max/e1>tol) && (n<epochs) && (rate > 0.00001)
        n=n+1
        ONN=NN
        (oc2tgrads,oc3tgrads)=(c2tgrads,c3tgrads)                               ##modification here
        NN.c2.b= NN.c2.b - rate*(c2tgrads.b)
        NN.c3.b= NN.c3.b - rate*(c3tgrads.b)
        NN.c2.w= NN.c2.w - rate*(c2tgrads.w)
        NN.c3.w= NN.c3.w - rate*(c3tgrads.w)
        (c2tgrads,c3tgrads,e1)=Find_Total_Gradients!(data_matrix,NN)
      
        if e1>e0
            NN=ONN
            rate=rate*.75
        (c2tgrads,c3tgrads,e1) = (oc2tgrads,oc3tgrads,e0)                 ##modification here 
            # e1=e0                                                                                   ##modification here
#           println(rate)
        elseif e1<e0
            rate=rate*1.05
            (oc2tgrads,oc3tgrads,e0)=(c2tgrads,c3tgrads,e1)
            max=maximum([maximum(abs.(c2tgrads.b)),maximum(abs.(c2tgrads.w)),maximum(abs.(c3tgrads.b)),maximum(abs.(c3tgrads.w))])
#           println(n,"   ",max,"  ",e1)
        end
    end
    println(n, " of max ",epochs, " bold epochs" )
#   toc()
    e1
end
######################################################################
######################################################################
function Train_Bold_Unbalanced!(inputdata_matrix::Array{Float64,2},outputdata_matrix::Array{Float64,2},UNN::UnbalancedNeuralNetwork,rate=.3,tol=0.0001,epochs=5000)
#   tic()

    (c2tgrads,c3tgrads,e0)=Find_Total_Gradients_Unbalanced!(inputdata_matrix,outputdata_matrix,UNN)
    max=1
    n=0
    e1=1.0
    if (UNN.nbottle != UNN.ncirc)
        rate=rate/UNN.outdim
    end
    
    while (max/e1>tol) && (n<epochs) && (rate > 0.00001)
        n=n+1
        ONN=UNN
        (oc2tgrads,oc3tgrads)=(c2tgrads,c3tgrads)                               ##modification here
        UNN.c2.b= UNN.c2.b - rate*(c2tgrads.b)
        UNN.c3.b= UNN.c3.b - rate*(c3tgrads.b)
        UNN.c2.w= UNN.c2.w - rate*(c2tgrads.w)
        UNN.c3.w= UNN.c3.w - rate*(c3tgrads.w)
        (c2tgrads,c3tgrads,e1)=Find_Total_Gradients_Unbalanced!(inputdata_matrix,outputdata_matrix,UNN)
      
        if e1>e0
            UNN=ONN
            rate=rate*.75
            (c2tgrads,c3tgrads,e1) = (oc2tgrads,oc3tgrads,e0)                 ##modification here 
            # e1=e0                                                                                   ##modification here
#           println(rate)
        elseif e1<e0
            rate=rate*1.05
            (oc2tgrads,oc3tgrads,e0)=(c2tgrads,c3tgrads,e1)
            max=maximum([maximum(abs.(c2tgrads.b)),maximum(abs.(c2tgrads.w)),maximum(abs.(c3tgrads.b)),maximum(abs.(c3tgrads.w))])
#           println(n,"   ",max,"  ",e1)
        end
    end
    println(n, " of max ",epochs, " bold epochs" )
#   toc()
    e1
end
######################################################################
function Train_Momentum_Stochastic!(data_matrix::Array{Float64,2},NN::NeuralNetwork,batchsize=10,rate=.3,momentum=.5,tol=0.0005,epochs=15000)
#   tic()
    oc2wt=zeros(size(NN.c2.w))
    oc3wt=zeros(size(NN.c3.w))
    oc2bt=zeros(size(NN.c2.b))
    oc3bt=zeros(size(NN.c3.b))

    o2changes=Layer_Connections(oc2wt,oc2bt)
    o3changes=Layer_Connections(oc3wt,oc3bt)
    new2changes=Layer_Connections(oc2wt,oc2bt)
    new3changes=Layer_Connections(oc3wt,oc3bt)
    (c2tgrads,c3tgrads,e0)=Find_Total_Gradients!(data_matrix,NN)
    
    if (NN.nbottle != NN.ncirc)
        rate=rate/NN.dim
    end    
    
    max=1
    n=0
    e1=1.0
    (rowsize,trainsize)=size(data_matrix);
    while (max/e1>tol) && (n<epochs)
        n=n+1
        new2changes.b = -rate*(c2tgrads.b) + momentum*o2changes.b
        new2changes.w = -rate*(c2tgrads.w) + momentum*o2changes.w
        new3changes.b = -rate*(c3tgrads.b) + momentum*o3changes.b
        new3changes.w = -rate*(c3tgrads.w) + momentum*o3changes.w

        NN.c2.b = NN.c2.b + new2changes.b
        NN.c2.w = NN.c2.w + new2changes.w
        NN.c3.b = NN.c3.b + new3changes.b
        NN.c3.w = NN.c3.w + new3changes.w

        e0=e1

        (c2tgrads,c3tgrads,e1)=Find_Total_Gradients!(data_matrix[:,sample(1:trainsize,batchsize,replace=false)],NN)

        o2changes.b = new2changes.b
        o2changes.w = new2changes.w
        o3changes.b = new3changes.b
        o3changes.w = new3changes.w

        max=maximum([maximum(abs.(c2tgrads.b)),maximum(abs.(c2tgrads.w)),maximum(abs.(c3tgrads.b)),maximum(abs.(c3tgrads.w))])
    end
    println(n, " of max ",epochs, " stochastic epochs" )
    
    n=0
    e1=1.0
    (rowsize,trainsize)=size(data_matrix);
    while (max/e1>tol) && (n<(epochs/3))
        n=n+1
        new2changes.b = -rate*(c2tgrads.b) + momentum*o2changes.b
        new2changes.w = -rate*(c2tgrads.w) + momentum*o2changes.w
        new3changes.b = -rate*(c3tgrads.b) + momentum*o3changes.b
        new3changes.w = -rate*(c3tgrads.w) + momentum*o3changes.w

        NN.c2.b = NN.c2.b + new2changes.b
        NN.c2.w = NN.c2.w + new2changes.w
        NN.c3.b = NN.c3.b + new3changes.b
        NN.c3.w = NN.c3.w + new3changes.w

        e0=e1

        (c2tgrads,c3tgrads,e1)=Find_Total_Gradients!(data_matrix,NN)

        o2changes.b = new2changes.b
        o2changes.w = new2changes.w
        o3changes.b = new3changes.b
        o3changes.w = new3changes.w

        max=maximum([maximum(abs.(c2tgrads.b)),maximum(abs.(c2tgrads.w)),maximum(abs.(c3tgrads.b)),maximum(abs.(c3tgrads.w))])
    end

    println(n, " of max ",epochs/3, " non-stochastic epochs" )
#   println("error $e1")
#   toc()
end

######################################################################
######################################################################
function Train_Momentum_Stochastic_Unbalanced!(inputdata_matrix::Array{Float64,2},outputdata_matrix::Array{Float64,2},UNN::UnbalancedNeuralNetwork,batchsize=10,rate=.3,momentum=.5,tol=0.0005,epochs=15000)
#   tic()
    oc2wt=zeros(size(UNN.c2.w))
    oc3wt=zeros(size(UNN.c3.w))
    oc2bt=zeros(size(UNN.c2.b))
    oc3bt=zeros(size(UNN.c3.b))

    o2changes=Layer_Connections(oc2wt,oc2bt)
    o3changes=Layer_Connections(oc3wt,oc3bt)
    new2changes=Layer_Connections(oc2wt,oc2bt)
    new3changes=Layer_Connections(oc3wt,oc3bt)
    (c2tgrads,c3tgrads,e0)=Find_Total_Gradients_Unbalanced!(inputdata_matrix,outputdata_matrix,UNN)
    
    if (UNN.nbottle != UNN.ncirc)
        rate=rate/UNN.outdim
    end    
    
    max=1
    n=0
    e1=1.0
    (irowsize,inputtrainsize)=size(inputdata_matrix);
    (orowsize,outputtrainsize)=size(outputdata_matrix);
    
    if (inputtrainsize != outputtrainsize)
        println("error")
    else
        trainsize=inputtrainsize   
    end

    while (max/e1>tol) && (n<epochs)
        n=n+1
        new2changes.b = -rate*(c2tgrads.b) + momentum*o2changes.b
        new2changes.w = -rate*(c2tgrads.w) + momentum*o2changes.w
        new3changes.b = -rate*(c3tgrads.b) + momentum*o3changes.b
        new3changes.w = -rate*(c3tgrads.w) + momentum*o3changes.w

        UNN.c2.b = UNN.c2.b + new2changes.b
        UNN.c2.w = UNN.c2.w + new2changes.w
        UNN.c3.b = UNN.c3.b + new3changes.b
        UNN.c3.w = UNN.c3.w + new3changes.w

        e0=e1

        batch=sample(1:trainsize,batchsize,replace=false)

        (c2tgrads,c3tgrads,e1)=Find_Total_Gradients_Unbalanced!(inputdata_matrix[:,batch],outputdata_matrix[:,batch],UNN)

        o2changes.b = new2changes.b
        o2changes.w = new2changes.w
        o3changes.b = new3changes.b
        o3changes.w = new3changes.w

        max=maximum([maximum(abs.(c2tgrads.b)),maximum(abs.(c2tgrads.w)),maximum(abs.(c3tgrads.b)),maximum(abs.(c3tgrads.w))])
    end
    println(n, " of max ",epochs, " stochastic epochs" )
    
    n=0
    e1=1.0
    
    (irowsize,inputtrainsize)=size(inputdata_matrix);
    (orowsize,outputtrainsize)=size(outputdata_matrix);
    
    while (max/e1>tol) && (n<(epochs/3))
        n=n+1
        new2changes.b = -rate*(c2tgrads.b) + momentum*o2changes.b
        new2changes.w = -rate*(c2tgrads.w) + momentum*o2changes.w
        new3changes.b = -rate*(c3tgrads.b) + momentum*o3changes.b
        new3changes.w = -rate*(c3tgrads.w) + momentum*o3changes.w

        UNN.c2.b = UNN.c2.b + new2changes.b
        UNN.c2.w = UNN.c2.w + new2changes.w
        UNN.c3.b = UNN.c3.b + new3changes.b
        UNN.c3.w = UNN.c3.w + new3changes.w

        e0=e1

        (c2tgrads,c3tgrads,e1)=Find_Total_Gradients_Unbalanced!(inputdata_matrix,outputdata_matrix,UNN)

        o2changes.b = new2changes.b
        o2changes.w = new2changes.w
        o3changes.b = new3changes.b
        o3changes.w = new3changes.w

        max=maximum([maximum(abs.(c2tgrads.b)),maximum(abs.(c2tgrads.w)),maximum(abs.(c3tgrads.b)),maximum(abs.(c3tgrads.w))])
    end

    println(n, " of max ",epochs/3, " non-stochastic epochs" )
#   println("error $e1")
#   toc()
end

######################################################################
######################################################################
function ExtractPhase(data_matrix::Array{Float64,2},NN::NeuralNetwork)
    points=size(data_matrix,2)
    phases=zeros(points)
    for n=1:points
        Feed_Forward!(data_matrix[:,n],NN)
        phases[n]=atan2(NN.blayer.a[1],NN.blayer.a[2])
    end
    phases
end
######################################################################
######################################################################
function ExtractMagnitude(data_matrix::Array{Float64,2},NN::NeuralNetwork)
    points=size(data_matrix,2)
    magnitudes=zeros(points)
    for n=1:points
        Feed_Forward!(data_matrix[:,n],NN)
        magnitudes[n]=(NN.blayer.z[1])^2+(NN.blayer.z[2])^2
    end
    magnitudes
end
######################################################################

######################################################################
######################################################################
function ExtractPhase_Unbalanced(inputdata_matrix::Array{Float64,2},UNN::UnbalancedNeuralNetwork)
    points=size(inputdata_matrix,2)
    phases=zeros(points)
    for n=1:points
        Feed_Forward_Unbalanced!(inputdata_matrix[:,n],UNN)
        phases[n]=atan2(UNN.blayer.a[1],UNN.blayer.a[2])
    end
    phases
end
######################################################################
######################################################################
function ExtractMagnitude_Unbalanced(inputdata_matrix::Array{Float64,2},UNN::UnbalancedNeuralNetwork)
    points=size(inputdata_matrix,2)
    magnitudes=zeros(points)
    for n=1:points
        Feed_Forward_Unbalanced!(inputdata_matrix[:,n],UNN)
        magnitudes[n]=(UNN.blayer.z[1])^2+(UNN.blayer.z[2])^2
    end
    magnitudes
end
######################################################################

######################################################################
######################################################################
function ExtractProjection(data_matrix::Array{Float64,2},NN::NeuralNetwork)
    points=size(data_matrix,2)
    coordinates=zeros(points,2)
    for n=1:points
        Feed_Forward!(data_matrix[:,n],NN)
        coordinates[n,:]=[NN.blayer.z[1],NN.blayer.z[2]]
    end
    coordinates
end

######################################################################
######################################################################
function CYCLOPS_Order(l_outs::Integer,l_norm_seed_data::Array{Float64,2},N_trials::Integer)
    scalefactor=1;
    besterror=10E20
    bestnet=0;
    NET=0;

    for trial=1:N_trials

        NET=Create_Network(l_outs,2,2);


        Train_Momentum_Stochastic!(l_norm_seed_data,NET);  
        global_circ_sse=Train_Bold!(l_norm_seed_data,NET);
     

        if global_circ_sse<besterror
            besterror=global_circ_sse
            bestnet=NET
        end
        print(global_circ_sse)
    end
    total_sse,onedlinear_sse=Default_Metrics(l_norm_seed_data,l_outs)
    besterror=besterror*2
    global_metrics=[besterror, -(total_sse-besterror)/besterror, -(onedlinear_sse-besterror)/besterror]
    
    estimated_phaselist=ExtractPhase(l_norm_seed_data,bestnet);
    estimated_phaselist=mod.(estimated_phaselist + 2*pi,2*pi)
    estimated_phaselist,bestnet,global_metrics
end 

######################################################################
function CYCLOPS_Order2(l_outs::Integer,l_norm_seed_data::Array{Float64,2},N_trials::Integer)
    scalefactor=1;
    besterror=10E20
    bestnet=0;
    NET=0;

    for trial=1:N_trials

#       NET=Create_Network(l_outs,2,2);
        NET=Create_Network(l_outs,3,2);


        Train_Momentum_Stochastic!(l_norm_seed_data,NET);  
        global_circ_sse=Train_Bold!(l_norm_seed_data,NET);
     

        if global_circ_sse<besterror
            besterror=global_circ_sse
            bestnet=NET
        end
        print(global_circ_sse)
    end
    total_sse,onedlinear_sse=Default_Metrics(l_norm_seed_data,l_outs)
    besterror=besterror*2
    global_metrics=[besterror, -(total_sse-besterror)/besterror, -(onedlinear_sse-besterror)/besterror]
    
    estimated_phaselist=ExtractPhase(l_norm_seed_data,bestnet);
    estimated_phaselist=mod.(estimated_phaselist + 2*pi,2*pi)
    estimated_phaselist,bestnet,global_metrics
end 
######################################################################
######################################################################
function CYCLOPS_Order_n(l_outs::Integer,l_norm_seed_data::Array{Float64,2},n_linear::Integer, N_trials::Integer)
    scalefactor=1;
    besterror=10E20
    bestnet=0;
    NET=0;

    for trial=1:N_trials

#       NET=Create_Network(l_outs,2,2);
        NET=Create_Network(l_outs,n_linear+2,2);


        Train_Momentum_Stochastic!(l_norm_seed_data,NET);  
        global_circ_sse=Train_Bold!(l_norm_seed_data,NET);
     

        if global_circ_sse<besterror
            besterror=global_circ_sse
            bestnet=NET
        end
        println(global_circ_sse)
    end
    total_sse,onedlinear_sse=Default_Metrics(l_norm_seed_data,l_outs)
    besterror=besterror*2
    global_metrics=[besterror, -(total_sse-besterror)/besterror, -(onedlinear_sse-besterror)/besterror]
    
    estimated_phaselist=ExtractPhase(l_norm_seed_data,bestnet);
    estimated_phaselist=mod.(estimated_phaselist + 2*pi,2*pi)
    estimated_phaselist,bestnet,global_metrics
end 
######################################################################
######################################################################


######################################################################
function CYCLOPS_Order_n_Unbalanced(insize::Integer,outsize::Integer,l_input_seed_data::Array{Float64,2},l_output_seed_data::Array{Float64,2},n_linear::Integer, N_trials::Integer)
    scalefactor=1;
    besterror=10E20
    bestnet=0;
    UNET=0;

    for trial=1:N_trials

        UNET=Create_Network_Unbalanced(insize,outsize,n_linear+2,2);


        Train_Momentum_Stochastic_Unbalanced!(l_input_seed_data,l_output_seed_data,UNET);  
        global_circ_sse=Train_Bold_Unbalanced!(l_input_seed_data,l_output_seed_data,UNET); 
     

        if global_circ_sse<besterror
            besterror=global_circ_sse
            bestnet=UNET
        end
        println(global_circ_sse)
    end
    total_sse,onedlinear_sse=Default_Metrics(l_output_seed_data,outsize)
    besterror=besterror*2
    global_metrics=[besterror, -(total_sse-besterror)/besterror, -(onedlinear_sse-besterror)/besterror]
    
    estimated_phaselist=ExtractPhase_Unbalanced(l_input_seed_data,bestnet);
    estimated_phaselist=mod.(estimated_phaselist + 2*pi,2*pi)
    estimated_phaselist,bestnet,global_metrics
end 
####################################################



#####################################################################
######################################################################
function CYCLOPS_Order_controlled(l_outs::Integer,l_norm_seed_data::Array{Float64,2},N_trials::Integer,batchsize=10,rate=.3,momentum=.5,tol=0.0005,epochs=50000)
    scalefactor=1;
    besterror=10E20
    bestnet=0;
    NET=0;

    for trial=1:N_trials

        NET=Create_Network(l_outs,2,2);

        Train_Momentum_Stochastic!(l_norm_seed_data,NET,batchsize,rate,momentum,tol,epochs); 
        Train_Momentum_Stochastic!(l_norm_seed_data,NET,size(l_norm_seed_data)[2],rate,momentum,tol/10,int(epochs/10));  
        global_circ_sse=Train_Bold!(l_norm_seed_data,NET);
     

        if global_circ_sse<besterror
            besterror=global_circ_sse
            bestnet=NET
        end
        print(global_circ_sse)
    end
    total_sse,onedlinear_sse=Default_Metrics(l_norm_seed_data,l_outs)
    besterror=besterror*2
    global_metrics=[besterror, -(total_sse-besterror)/besterror, -(onedlinear_sse-besterror)/besterror]
    
    estimated_phaselist=ExtractPhase(l_norm_seed_data,bestnet);
    estimated_phaselist=mod.(estimated_phaselist + 2*pi,2*pi)
    estimated_phaselist,bestnet,global_metrics
end 
######################################################################
######################################################################
function Default_Metrics(data::Array{Float64,2},outsize::Integer)
    totalsse=0
  
    Mod=fit(PCA,data,method=:svd,pratio=.99999999999999999999999999999)
    data=transform(Mod,data)
    usesize=size(data)[1]
    for gene=1:usesize
        totalsse=totalsse+var(data[gene,:])
    end
    linearsse=totalsse-var(data[1,:])
    totalsse,linearsse
end
######################################################################
######################################################################
######################################################################
######################################################################
function CYCLOPS_Order_base(l_outs::Integer,l_norm_seed_data::Array{Float64,2},N_trials::Integer,NN::NeuralNetwork)
    scalefactor=1;
    besterror=10E20
    bestnet=0;
    NET=0;

    for trial=1:N_trials

        NET=NN;

        Train_Momentum_Stochastic!(l_norm_seed_data,NET);  
        global_circ_sse=Train_Bold!(l_norm_seed_data,NET);
     

        if global_circ_sse<besterror
            besterror=global_circ_sse
            bestnet=NET
        end
        print(global_circ_sse)
    end
    total_sse,onedlinear_sse=Default_Metrics(l_norm_seed_data,l_outs)
    besterror=besterror*2
    global_metrics=[besterror, -(total_sse-besterror)/besterror, -(onedlinear_sse-besterror)/besterror]

    estimated_phaselist=ExtractPhase(l_norm_seed_data,bestnet);
    estimated_phaselist=mod.(estimated_phaselist + 2*pi,2*pi)
    estimated_phaselist,bestnet,global_metrics
end 
######################################################################
######################################################################
end
