include("EnergyModels.jl")
using .EnergyModels
using ArgParse
using Dates
using Flux
using MLDatasets: MNIST, CIFAR10

s = ArgParseSettings(autofix_names = true)
@add_arg_table! s begin
    "--model"
        help = "model"
        default = "CNN"
    "--task"
        help = "which datset (CIFAR10 or MNIST)"
        default = "CIFAR10"
    "--archi"
        help = "architecture for MLP (number of neurons by layer)"
        default = [784, 512, 10]
        nargs = '*'
        arg_type = Int
    "--channels"
        help = "number of channels in each layer for a CNN"
        default = [128, 256, 512]
        nargs = '*'
        arg_type = Int
    "--kernels"
        help = "convolutional kernel width in each layer for a CNN"
        default = [3, 3, 3]
        nargs = '*'
        arg_type = Int
    "--paddings"
        help = "padding in each layer for a CNN"
        default = [0, 0, 0]
        nargs = '*'
        arg_type = Int
    "--strides"
        help = "convolutional stride in each layer for a CNN"
        default = [1, 1, 1]
        nargs = '*'
        arg_type = Int
    "--fc"
        help = "number of neurons in fully connected portion after convolutional layers"
        default = []
        nargs = '*'
        arg_type = Int
    "--lrs"
        help = "learning rates per layer"
        default = [5f-4, 4f-4]
        nargs = '*'
        arg_type = Float32
    "--act"
        help = "activation function"
        default = "mysig"
    "--alg"
        help = "training algorithim (EP, CEP, BPTT)"
        default = "EP"
    "--loss"
        help = "loss type (cross entropy 'cel' or mean square error 'mse')"
        default = "cel"
    "--mbs"
        help = "minibatch size"
        default = 100
        arg_type = Int
    "--T1"
        help = "number of timesteps of first phase"
        default = 100
        arg_type = Int
    "--T2"
        help = "number of timesteps of second phase"
        default = 10
        arg_type = Int
    "--dt"
        help = "duration of timestep"
        default = 1.0f-2
        arg_type = Float32
    "--betas"
        help = "betas (nudging strength. First value should be 0 for free phase)"
        default = [0.0f0, 0.01f0]
        nargs = 2
        arg_type = Float32
    "--epochs"
        help = "number of epochs to train for"
        default = 1
        arg_type = Int
    "--random-sign"
        help = "run in random sign mode (randomly flip beta of phase 2)"
        action = :store_true
    "--thirdphase"
        help = "whether to run another clamped phase with beta of opposite sign"
        action = :store_true
    "--save"
        help = "save the model to disk?"
        action = :store_true
    "--todo"
        help = "'train', ()"
        default = "train"
    "--load-path"
        help = "path to a previously saved model to load"
        default = ""
    "--device"
        help = "device"
        default = "gpu"
end

argdict = parse_args(s)
args = (; (Symbol(k) => v for (k,v) in argdict)...)

command_line = join(ARGS, " ")

println("\n")
println(command_line)
println("\n")
println("##################################################################")
println("\nargs\tmbs\tT1\tT2\tepochs\tactivation\tbetas")
println("\t",args.mbs,"\t",args.T1,"\t",args.T2,"\t",args.epochs,"\t",args.act, "\t", args.betas)
println("\n")

if args.save
    date = string(Date(Dates.now()))
    time = string(Time(Dates.now()))
    if args.load_path==""
        path = "results/"*args.alg*"/"*args.loss*"/"*date*"/"*time*"_gpu"*string(args.device)
    else
        path = args.load_path
    end
    mkpath(path)
else
    path = ""
end

mbs = args.mbs

if args.device == "gpu"
    device = Flux.gpu
elseif args.device == "cpu"
    device = Flux.cpu
else
    throw(error("invalid device specification $(args.device) (should be either 'gpu' or 'cpu')"))
end


# Activation functions
function my_sigmoid(x)
    return 1/(1+exp(-4*(x-0.5)))
end
function hard_sigmoid(x)
    return (1+Flux.hardtanh(2*x-1))*0.5
end
function ctrd_hard_sig(x)
    return (Flux.hardtanh(2*x))*0.5
end
function my_hard_sig(x)
    return (1+Flux.hardtanh(x-1))*0.5
end

if args.act=="mysig"
    activation = my_sigmoid
elseif args.act=="sigmoid"
    activation = sigmoid
elseif args.act=="tanh"
    activation = tanh
elseif args.act=="hard_sigmoid"
    activation = hard_sigmoid
elseif args.act=="my_hard_sig"
    activation = my_hard_sig
elseif args.act=="ctrd_hard_sig"
    activation = ctrd_hard_sig
elseif args.act == "relu"
    activation = Flux.relu
end

if args.task == "MNIST" || args.task == "CIFAR10"
    nlabels = 10
end

if args.task == "MNIST"
    Dataset = MNIST
    channels = vcat([1], args.channels)
    insize = 28
elseif args.task == "CIFAR10"
    Dataset = CIFAR10
    channels = vcat([3], args.channels) # rgb
    insize = 32
else
    throw(error("invalid dataset $(args.dataset)"))
end

kernels = args.kernels
@assert length(kernels) == length(channels)-1 "number of channels does not match number of kernels"

if args.load_path == ""
    if args.model == "MLP"
        # archi = vcat(nlabels, args.archi)
        archi = args.archi
        synapses = collect([Dense(archi[i] => archi[i+1], activation, bias=true, init=Flux.glorot_uniform) for i in 1:length(archi)-1])
    elseif args.model == "CNN"
        synapses = [] # collect([Conv((kernels[i], kernels[i]), channels[i] => channels[i+1], activation, bias=true, init=Flux.glorot_uniform) for i in eachindex(kernels)]) 
        size = insize
        archi = [(insize, insize, channels[1])]
        for i in eachindex(kernels)
            append!(synapses, [Conv((kernels[i], kernels[i]), channels[i] => channels[i+1], pad=args.paddings[i], stride=args.strides[i], init=Flux.glorot_uniform, bias=false)])
            global size = floor( ( size + 2*args.paddings[i] - kernels[i] + 1 ) / args.strides[i] )
            append!(archi, [(size, size, channels[i+1])])
        end
        size = size*size*channels[end]
        for idx in eachindex(args.fc)
            append!(synapses, [Dense(size => args.fc[idx], activation, bias=true)])
            global size = args.fc[idx]
            append!(archi, [size])
        end
    end
else
    save_state = JLD2.load(args.load_path * "/checkpoint.jld2", "model_state")
    archi = save_state.archi
    synapses = save_state.synapses
    optimizer = save_state.opt
end

if args.todo == "train"
    if args.model == "MLP"
        transform = x -> reshape(x, (:, size(x)[end])) 
    elseif args.model == "CNN"
        #transform = x -> reshape(x, (size(x)[1:2]..., channels[1], size(x)[end]))
    end
    train_loader = Flux.BatchView(Flux.shuffleobs(Dataset(split=:train)), batchsize=mbs)
    test_loader  = Flux.BatchView(Flux.shuffleobs(Dataset(split=:test )), batchsize=mbs)

    layeroptimizers = []
    for idx in 1:length(synapses)
        append!(layeroptimizers, [Flux.setup(Flux.Adam(args.lrs[idx]), synapses[idx])])
    end

    println(string(layeroptimizers)[1:300], "...")
    if args.alg == "EP"
        EnergyModels.trainEP(archi, synapses, layeroptimizers, train_loader, test_loader, args.T1, args.T2, args.betas, device, args.epochs,
            random_sign=args.random_sign, save=args.save, path=path, thirdphase=args.thirdphase)
    elseif args.alg == "LCA"
        EnergyModels.trainLCA!(archi, args, synapses, layeroptimizers, train_loader, test_loader, device, path=path)
    end
end
