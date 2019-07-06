mutable struct NN <: Functor
    atomembeds
    countembeds
    hsize
    l_h
    l_out
    l_out2
end

function NN()
    T = Float32
    atomembeds = Uniform(-0.01,0.01)(T, 50, 7) |> parameter
    countembeds = Uniform(-0.01,0.01)(T, 50, 50) |> parameter
    hsize = 100
    asize = size(atomembeds, 1)
    csize = size(countembeds, 1)
    # l_a = Linear(T, asize, hsize)
    l_h = Linear(T, 2(asize+hsize)+csize+1, 2hsize)
    l_out = Linear(T, hsize, 1)
    l_out2 = Linear(T, hsize, hsize)
    NN(atomembeds, countembeds, hsize, l_h, l_out, l_out2)
end

function (nn::NN)(data::Vector, indexes::Vector{Int})
    x = Sample(data, indexes)
    a = lookup(nn.atomembeds, x.a)
    c = lookup(nn.countembeds, x.c)
    c = expand(c, x.dims_c)
    h = zero(a, nn.hsize, size(a,2))
    for i = 1:5
        h = concat(1, a, h)
        h = lookup(h, x.t)
        h = concat(1, h, c, x.d)
        h = nn.l_h(h)
        h = gate(h)
        h = average(h, x.dims_t)
    end
    h = average(h, x.dims_a)
    h = nn.l_out2(h)
    h = relu(h)
    o = nn.l_out(h)
    if Merlin.istraining()
        mse(x.y, o)
    else
        Array(x.y.data), Array(o.data)
    end
end

function gate(x::Var)
    n = size(x,1) ÷ 2
    a = tanh(x[1:n,:])
    b = sigmoid(x[n+1:2n,:])
    a .* b
end

function expand(x::Var, dims)
    ys = Var[]
    for i = 1:length(dims)
        d = dims[i]
        y = repeat(x[:,i:i], 1, d)
        push!(ys, y)
    end
    concat(2, ys...)
end
