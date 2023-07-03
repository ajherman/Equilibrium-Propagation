using Flux
using MLDatsets: CIFAR10, MNIST
using Base.Iterators: partition
using Dates
using Random
using JLD2

function energymodel(archi, x, synapses, beta, y, T)
    neurons = zeros(archi)
    ascendphi = Flux.setup(Flux.Descent(1), neurons)

    for t in 1:T
        phi = 0
        
        phi, grads = Flux.widthgradient(neurons) do
            phi += synapses[1](x) * neurons[1]
            for idx in 2:length(synapses)
                phi += sum( synapses[idx](neurons[idx-1]) * neurons[idx], dims=1 ) # sum across neuron dimension not layer or batch
                # neuron_tendencies = 
            end
            phi -= beta*Flux.crossentropy(y, neurons[end])
            return phi
        end
        
        Flux.update!(ascendphi, neurons, neurongrads)
    end

    return neurons
end



function trainEP(archi, synapses, optimizer, train_loader, test_loader, T1, T2, betas, device, epochs,
                        random_sign=false, save=false, path="", checkpoint=nothing, thirdphase=false)
    
    mbs = size(train_loader)[end]
    iter_per_epochs = length(train_loader)
    starttime = Dates.now()

    if checkpoint is nothing
        train_acc = [0.0]
        test_acc = [0.0]
        best = 0.0
        epoch_sofar = 0
     else
        train_acc   = checkpoint["train_acc"]
        test_acc    = checkpoint["test_acc"]
        best        = checkpoint["best"]
        epoch_sofar = checkpoint["epoch"]
    end

    train_loader = train_loader |> device
    test_loader = test_loader |> device
    synapses = synapses |> device
    for epoch in 1:epochs
        run_correct = 0
        run_total = 0

        for (idx, (x, y)) in enumerate(train_loader)
            neurons = zeros(archi) |> device

            # first phase
            neurons = energymodel(neurons, x, y, synapses, betas[1], T1)
            neurons_1 = copy(neurons)

            # measure accuracy
            pred = Flux.onecold(neurons_1[end])
            run_correct += sum(All(pred .== y))
            run_total += size(y)[end]
            
            # second phase
            if random_sign
                sgn = 2*Random.bitrand(1) - 1
            else
                sgn = 1
            end

            neurons = energymodel(neurons_1, x, y, synapses, betas[2]*sgn, T2)
            neurons_2 = copy(neurons)

            # third phase 
            if third_phase
                neurons = copy(neurons_1)
                neurons = energymodel(neurons, x, y, synapses, -betas[2], T2)
                neurons_3 = copy(neurons)
                
                grads = compute_syn_grads(neurons_2, neurons_3, x, y, synapses, betas[2], -betas[2])
            else
                grads = compute_syn_grads(neurons_1, neurons_2, x, y, synapses, betas[1], betas[2])
            end
            
            # update weights
            Flux.update!(optimizer, synapses, grads)
            
            # print progress
            if mod(idx, iter_per_epochs/10) == 0 || idx == iter_per_epochs-1
                run_acc = run_correct / run_total
                timesince = Dates.now() - starttime
                remaining = timesince * (epochs*iter_per_epochs - idx - epoch*iter_per_epochs) / (epoch*iter_per_epochs + idx)
                println("Epoch : $(epoch_sofar + epoch + idx/iter_per_epochs)")
                println("\tRun train acc : $(run_acc)\t($(run_correct)/$(run_total))")
                println("\tElapsed time : $(timesince) \t ETA $(remaining)")
            end
            
        end

        # testing
        test_correct = 0
        for (x, y) in test_loader
            neurons = zeros(archi)
            neurons = energymodel(neurons, x, y, synapses, 0.0, T1)
            test_correct += sum(All(Flux.onecold(neurons[end]) .== y))
        end
        test_acc_t = test_correct/(length(test_loader))
        println("Test acc : $(test_acc_t)")

        # save to file
        if save
            append!(test_acc, test_acc_t)
            append!(run_acc, train_acc)
            if test_acc_t > best
                best = test_acc_t
                save_dic = (archi = archi, synapses = Flux.state(synapses), opt = Flux.state(optimizer),
                                  train_acc = train_acc, test_acc = test_acc, best = best,
                                  epoch = epoch_sofar + epoch)
                jldsave(path * "/checkpoint.jld2", model_state=save_dic)
            end
        end

    end

    # save final model
    if save
        save_dic = (archi = archi, synapses = Flux.state(synapses), opt = Flux.state(optimizer),
                          train_acc = train_acc, test_acc = test_acc, best = best,
                          epoch = epoch_sofar + epoch)
        jldsave(path * "/final.jld2", model_state=save_dic)
    end

end
