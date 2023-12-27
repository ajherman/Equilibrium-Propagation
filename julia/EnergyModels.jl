module EnergyModels
using CUDA
using Flux
using Dates
using Random
using JLD2

function Phi(synapses, x, y, neurons, beta)
    phi = 0.0f0
    layers = vcat([x], neurons)
    for idx in eachindex(synapses)
        phi += sum( synapses[idx](layers[idx]) .* layers[idx+1] ) # sum across neuron dimension not layer
    end
    # phi -= beta*Flux.crossentropy(neurons[end], y)
    phi -= beta*Flux.mse(neurons[end], y)
    return phi
end

function energymodelimplicit(synapses, x, y, neurons, T, beta)
    ascendphi = Flux.setup(Flux.Descent(1), neurons)

    for t in 1:T
        
        neurongrads = Flux.gradient(neurons) do n
            Phi(synapses, x, y, n, beta)
        end
        
        # Flux.update!(ascendphi, neurons, neurongrads[1])
        neurons[:] = neurongrads[1][:]
        neurons = Flux.sigmoid(neurons)
    end

    return neurons
end

function energymodelanalytical(synapses, x, y, neurons, T, beta)
    layers = vcat([x], neurons)
    dn = collect([zero(n) for n in neurons]) 
    for t in 1:T
        for idx in eachindex(layers)[2:end-1]
            dn[idx-1] = synapses[idx-1](layers[idx-1])
            dn[idx-1] += synapses[idx].weight' * (layers[idx+1] .-synapses[idx].bias)
        end
        dn[end] = synapses[end](layers[end-1])
        nudge = beta * (y - layers[end]) # analytical gradient of MSE(y, pred)
        dn[end] += nudge
        for idx in eachindex(dn)
            dn[idx] -= layers[idx+1] # exponential decay
            layers[idx+1] = Flux.sigmoid(layers[idx+1] + dn[idx])
        end
    end

    return layers[2:end]
end


function compute_syn_grads(synapses, x, y, neurons_1, neurons_2, beta_1, beta_2)
    # Computing the EP update given two steady states neurons_1 and neurons_2, static input x, label y
    syn_grads = Flux.gradient(synapses) do syn
        phi_1 = Phi(syn, x, y, neurons_1, beta_1)
        
        phi_2 = Phi(syn, x, y, neurons_2, beta_2)
        
        delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)        
        return delta_phi # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem
    end

    return syn_grads[1]
end


function trainEP(archi, synapses, optimizer, train_loader, test_loader, T1, T2, betas, device, epochs,
                        ; random_sign=false, save=false, path="", checkpoint=nothing, thirdphase=false)
    synapses = synapses |> device


    # energymodel = energymodelimplicit
    energymodel = energymodelanalytical
    
    mbs = Flux.batchsize(train_loader)
    # mbs = size(first(train_loader)[1])[end]
    iter_per_epochs = length(train_loader)
    starttime = Dates.now()

    if checkpoint === nothing
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

    labels = 0:(archi[end]-1)

    train_loader = train_loader |> device
    test_loader = test_loader |> device
    synapses = synapses |> device
    optimizer = optimizer .|> device
    initneurons = collect([zeros(n, mbs) for n in archi[2:end]]) |> device
    neurons_1 = collect([zeros(n, mbs) for n in archi[2:end]]) |> device
    neurons_2 = collect([zeros(n, mbs) for n in archi[2:end]]) |> device
    if thirdphase
        neurons_3 = collect([zeros(n, mbs) for n in archi[2:end]]) |> device
    end
    neurons = collect([zeros(n, mbs) for n in archi[2:end]]) |> device
    for epoch in 1:epochs
        run_correct = 0
        run_total = 0

        for (idx, (x, y)) in enumerate(train_loader)
            x = reshape(x, (:, size(x)[end])) # flatten image input for MLP
            x = x |> device
            y = float.(Flux.onehotbatch(y, labels)) |> device
            neurons[:] .= initneurons[:]

            # first phase
            neurons = energymodel(synapses, x, y, neurons, T1, betas[1])
            neurons_1[:] .= neurons[:]

            # measure accuracy
            pred = Flux.onecold(neurons_1[end])
            label = Flux.onecold(y)
            run_correct += sum((pred .== label))
            run_total += size(y)[end]
            
            # second phase
            if random_sign
                sgn = 2*Random.bitrand(1)[1] - 1
            else
                sgn = 1
            end

            neurons = energymodel(synapses, x, y, neurons, T2, betas[2]*sgn)

            # third phase 
            if thirdphase
                neurons_2[:] .= neurons[:]
                neurons[:] .= neurons_1[:]
                neurons = energymodel(synapses, x, y, neurons, T2, -betas[2])
                neurons_3 = copy(neurons)
                
                grads = compute_syn_grads(synapses, x, y, neurons_2, neurons_3, betas[2], -betas[2])
            else
                grads = compute_syn_grads(synapses, x, y, neurons_1, neurons, betas[1], betas[2])
            end
            
            # update weights
            Flux.update!.(optimizer, synapses, grads)
            
            # print progress
            if mod(idx, round(iter_per_epochs/10)) == 0
                run_acc = run_correct / run_total
                timesince = Dates.now() - starttime
                percent = (idx + (epoch-1)*iter_per_epochs) / (epochs*iter_per_epochs)
                remaining = (Dates.Nanosecond(timesince).value * percent * (epochs*iter_per_epochs - idx - (epoch-1)*iter_per_epochs))
                remaining = Dates.canonicalize(Dates.Nanosecond(round(remaining)))
                println("Epoch : $(epoch_sofar + epoch-1 + idx/iter_per_epochs) of $(epochs)")
                println("\tRun train acc : $(run_acc)\t($(run_correct)/$(run_total))")
                println("\tElapsed time : $(Dates.canonicalize(timesince)) \t$(percent*100)%\t ETA $(remaining)")
            end
            
        end

        # testing
        test_correct = 0
        for (x, y) in test_loader
            x = x |> device
            y = float.(Flux.onehotbatch(y, labels)) |> device
            neurons = copy(initneurons)
            neurons = energymodel(synapses, x, y, neurons, T1, 0.0)
            pred = Flux.onecold(neurons[end])
            label = Flux.onecold(y)
            test_correct += sum((pred .== label))
        end
        test_acc_t = test_correct/(length(test_loader))/mbs
        println("Test acc : $(test_acc_t)")

        # save to file
        if save
            append!(test_acc, test_acc_t)
            append!(train_acc, run_correct/run_total)
            if test_acc_t > best
                best = test_acc_t
                save_dic = (archi = archi, synapses = Flux.state(synapses), opt = Flux.state.(optimizer),
                                  train_acc = train_acc, test_acc = test_acc, best = best,
                                  epoch = epoch_sofar + epoch)
                jldsave(path * "/checkpoint.jld2", model_state=save_dic)
            end
        end

    end

    # save final model
    if save
        save_dic = (archi = archi, synapses = Flux.state(synapses), opt = Flux.state.(optimizer),
                          train_acc = train_acc, test_acc = test_acc, best = best,
                          epoch = epoch_sofar + epochs)
        jldsave(path * "/final.jld2", model_state=save_dic)
    end

end




function lca!(x, xhat, memp, neurons, W, WT, T, dt=0.01)
    for t in 1:T
        xhat = WT(neurons[1])
        memp += dt*(W(x-xhat) - neurons[1] - memp)
        neurons[1] = activation(memp)
    end
    xhat = WT(neurons[1])
end

function compute_lca_syn_grads(synapses, x, xhat, neurons)
    syn_grads = Flux.gradient(synapses) do syn
        return syn[1](x - xhat)*neurons[1] # W grad = (representation y outer product (x-xhat)) ----> d/dW ( W[x - xhat] * y ) 
    end

    return syn_grads[1]
end


function postupdate!(W, WT)
    # load forward weights into WT
    s = Flux.state(WT)
    snew = (s..., weight=Flux.transpose(W.weight, (4,3)))
    Flux.loadmodel!(WT, snew)
end

function trainLCA!(archi, args, synapses, layeroptimizers, train_loader, test_loader, device; path=path, checkpoint=nothing)
    
    
    mbs = Flux.batchsize(train_loader)
    # mbs = size(first(train_loader)[1])[end]
    iter_per_epochs = length(train_loader)
    starttime = Dates.now()

    if checkpoint === nothing
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
    layeroptimizers = layeroptimizers .|> device
    initneurons = collect([zeros(n..., mbs) for n in archi[2:end]]) |> device
    #neurons_1 = collect([zeros(n..., mbs) for n in archi[2:end]]) |> device
    #neurons_2 = collect([zeros(n..., mbs) for n in archi[2:end]]) |> device
    neurons = collect([zeros(n..., mbs) for n in archi[2:end]]) |> device
    memp = collect([zeros(n..., mbs) for n in archi[2:end]]) |> device
    xhat = zeros(archi[1]..., mbs)

    W = synapses[1]
    WT = Flux.ConvTranspose(size(W.weight)[1:2], size(W.weight)[3] => size(W.weight)[4], identity, pad=W.pad, stride=W.stride, bias=false, init=zeros)

    postupdate!(W, WT)

    for epoch in 1:args.epochs
        run_correct = 0
        run_total = 0

        for (idx, (x, y)) in enumerate(train_loader)
            #x = reshape(x, (:, size(x)[end])) # flatten image input for MLP
            x = x |> device

            neurons[:] .= initneurons[:]
            
            lca!(x, xhat, memp, neurons, synapses[1], args.T1, args.dt)
            
            syn_grads = compute_lca_syn_grads(synapses, x, xhat, neurons)

            # update weights
            Flux.update!.(layeroptimizer, synapses, syn_grads)
            postupdate!(synapses[1], WT)

            # print progress
            if mod(idx, round(iter_per_epochs/10)) == 0
                recon_err = sum((x-xhat)^2)
                timesince = Dates.now() - starttime
                percent = (idx + (epoch-1)*iter_per_epochs) / (epochs*iter_per_epochs)
                remaining = (Dates.Nanosecond(timesince).value * percent * (epochs*iter_per_epochs - idx - (epoch-1)*iter_per_epochs))
                remaining = Dates.canonicalize(Dates.Nanosecond(round(remaining)))
                println("Epoch : $(epoch_sofar + epoch-1 + idx/iter_per_epochs) of $(epochs)")
                println("\tReconstruction error : $(recon_err)")
                println("\tElapsed time : $(Dates.canonicalize(timesince)) \t$(percent*100)%\t ETA $(remaining)")
            end
        end

        # save to file
        if args.save
            append!(recon_errs, recon_err)
            if recon_err < best
                best = recon_err
                save_dic = (args=args, synapses = Flux.state(synapses), opt = Flux.state.(layeroptimizers),
                                  recon_err=recon_err, best = best,
                                  epoch = epoch_sofar + epoch)
                jldsave(path * "/checkpoint.jld2", model_state=save_dic)
            end
        end

    end

    # save final model
    if save
        save_dic = (args=args, synapses = Flux.state(synapses), opt = Flux.state.(layeroptimizers),
                          recon_err=recon_err, best = best,
                          epoch = epoch_sofar + epoch)
        jldsave(path * "/final.jld2", model_state=save_dic)
    end
end
end
