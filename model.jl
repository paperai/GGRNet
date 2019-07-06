using Random

mutable struct Model
    config
    transformer
    nn
end

function Model(config::Dict)
    # filename = config["train_file"]
    # data = readdata(filename[1:end-4] * ".sdf")
    filename = ".data/gdb9.sdf"
    data = readdata(filename)
    n = length(data)
    r = randperm(n)
    # m = trunc(Int, n*0.1)
    m = 10000
    devdata = data[r[1:m]]
    testdata = data[r[m+1:2m]]
    traindata = data[r[2m+1:7m]]
    # traindata = data[r[2m+1:end]]

    train_y = map(x -> x.y, traindata)
    transformer = Standardizer(train_y)
    trans_y = transformer(map(x -> x.y, data))
    for i = 1:length(data)
        data[i].y = trans_y[i]
    end

    nn = NN()
    @info "# Train:\t$(length(traindata))"
    @info "# Dev:\t$(length(devdata))"
    @info "# Test:\t$(length(testdata))"
    m = Model(config, transformer, nn)
    train!(m, traindata, devdata, testdata)
    m
end

function train!(model::Model, traindata, devdata, testdata)
    config = model.config
    opt = SGD(clip=10.0)
    # lr = config["learning_rate"]
    lr = 0.01
    nn = todevice(model.nn)
    batchsize = config["batchsize"]
    # batchsize = 20
    mindev, mintest = typemax(Float64), typemax(Float64)

    nepochs = config["nepochs"]
    for epoch = 1:100
        println("Epoch:\t$epoch")
        opt.rate = lr / (1 + 0.01*(epoch-1))

        loss = minimize!(nn, traindata, opt, batchsize=batchsize, shuffle=true)
        loss /= length(traindata)
        println("Loss:\t$loss")

        println("-----Test data-----")
        res = evaluate(nn, testdata, batchsize=50)
        testscore = mae(res, model.transformer)
        # writefile("pred.txt", res, model.transformer)
        println("-----Dev data-----")
        res = evaluate(nn, devdata, batchsize=50)
        devscore = mae(res, model.transformer)
        if devscore <= mindev
            mindev = devscore
            mintest = testscore
        end
        println("-----Final test-----")
        println("Dev: $mindev")
        println("Test: $mintest")
        println()
    end
end

function mae(data::Vector, trans)
    data1 = collect(Iterators.flatten(map(x -> vec(x[1]), data)))
    data1 = reshape(data1, 1, length(data1))
    data1 = inverse(trans, data1)
    data2 = collect(Iterators.flatten(map(x -> vec(x[2]), data)))
    data2 = reshape(data2, 1, length(data2))
    data2 = inverse(trans, data2)
    data = abs.(data1 - data2)
    score = sum(data) / length(data)
    println("MAE:\t$score")
    score
end

function writefile(filename::String, data::Vector, trans)
    data1 = collect(Iterators.flatten(map(x -> vec(x[1]), data)))
    data1 = inverse(trans, data1)
    data2 = collect(Iterators.flatten(map(x -> vec(x[2]), data)))
    data2 = inverse(trans, data2)
    open(filename,"w") do io
        for (y,z) in zip(data1,data2)
            println(io, "$y\t$z")
        end
    end
end
