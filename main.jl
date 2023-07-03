using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "--model"
        help = "model"
        default = "MLP"
    "--task"
        help = "which datset (CIFAR10 or MNIST)"
        default = "MNIST"
    "--archi"
        help = "architecture"
        default = [784, 512, 10]
    "--act"
        help = "activation function"
        default = "mysig"
    "--mbs"
        help = "minibatch size"
        default = 100
    "--T1"
        help = "duration of first phase"
        default = 100
    "--T2"
        help = "duration of second phase"
        default = 10
    "--betas"
        help = "betas (nudging strength. First value should be 0 for free phase)"
        default = [0.0, 0.01]
    "--epochs"
        help = "number of epochs to train for"
        default = 1
    "--random-sign"
        help = "run in random sign mode (randomly flip beta of phase 2)"
        default = false
        action = :store_true
    "--save"
        help = "save the model to disk?"
        default = true
    "--todo"
        help = "'train', ()"
        default = "train"
    "--load-path"
        help = "path to a previously saved model to load"
        default = ""
    "--device"
        help = "device"
        default = 0
end

args = parse_args(s)

command_line = join(ARGS, " ")

println("\n")
println(command_line)
println("\n")
println("##################################################################")
println("\nargs\tmbs\tT1\tT2\tepochs\tactivation\tbetas")
println("\t",args.mbs,"\t",args.T1,"\t",args.T2,"\t",args.epochs,"\t",args.act, "\t", args.betas)
println("\n")

if args.save
    date = Dates.now()
    time = Dates.now()
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

device = Flux[args.device]

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
end

if args.dataset == "MNIST" || args.dataset == "CIFAR10"
    nlabels = 10
end

if args.load_path == ""
    if args.model == "MLP"
        archi = hcat(nlabels, args.archi)
        synapses = collect([Dense(archi[i] => archi[i+1], σ=activation, bias=true, init=glorot_uniform) for i in 1:len(archi)-1])
    end
    optimizer = Flux.Adam(0.01)
else
    save_state = JLD2.load(args.load_path * "/checkpoint.jld2", "model_state")
    archi = save_state.archi
    synapses = save_state.synapses
    optimizer = save_state.opt
end

synapses = synapses |> device


if args.todo == "train"
    Dataset = Flux[args.dataset]
    train_loader = Dataset.traindata(Float32)
    test_loader = Dataset.testdata(Float32)

    println(optimizer)
    trainEP(archi, synapses, optimizer, train_loader, test_loader, args.T1, args.T2, args.betas, device,
                   args.epochs, ags.random_sign, args.save, path, thirdphase=args.thirdphase)
end
