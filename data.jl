mutable struct Sample
    a
    dims_a
    t
    dims_t
    d
    c
    dims_c
    y
end

function Sample(samples::Vector{Sample}, indexes::Vector{Int})
    data = samples[indexes]
    a = map(x -> x.a, data)
    dims_a = length.(a)
    a = cat(a..., dims=1)
    a = reshape(a, 1, length(a)) |> todevice

    offs = 0
    i = 1
    t = map(data) do x
        t = x.t .+ offs
        offs += dims_a[i]
        i += 1
        t
    end
    t = cat(t..., dims=2) |> todevice
    dims_t = cat(map(x -> x.dims_t, data)..., dims=1)

    d = cat(map(x -> x.d, data)..., dims=1)
    d = reshape(d, 1, length(d)) |> todevice
    c = reshape(dims_a, 1, length(dims_a)) |> todevice
    dims_c = cat(map(x -> x.dims_c, data)..., dims=1)

    y = map(x -> x.y, data)
    y = reshape(y, 1, length(y)) |> todevice
    Sample(Var(a), dims_a, Var(t), dims_t, Var(d), Var(c), dims_c, Var(y))
end

function readdata(filename::String)
    mols = readsdf(filename)
    data = Sample[]
    atomdict = Dict("H"=>1, "C"=>2, "N"=>3, "O"=>4, "F"=>5, "S"=>6, "Cl"=>7)
    for m in mols
        m = complete(m)
        props = split(m["props"], "\t", keepempty=false)
        m["props"] = [parse(Float32,props[i]) for i=5:length(props)]
        # m["props"] = [parse(Float32,props[i]) for i=1:length(props)]
        y = m["props"][3]

        atomids = map(a -> atomdict[a.symbol], m.atoms)
        tails = [Int[] for _=1:natoms(m)]
        dists = [Float32[] for _=1:natoms(m)]
        for k = 1:length(m.bonds)
            b = m.bonds[k]
            push!(tails[b.i], b.i, b.j)
            push!(tails[b.j], b.j, b.i)
            a1 = m.atoms[b.i]
            a2 = m.atoms[b.j]
            dist = (a1.x-a2.x)^2 + (a1.y-a2.y)^2 + (a1.z-a2.z)^2
            dist = Float32(1/sqrt(dist))
            push!(dists[b.i], dist)
            push!(dists[b.j], dist)
        end
        dims_t = map(x -> length(x)÷2, tails)
        tails = collect(Iterators.flatten(tails))
        tails = reshape(tails, 2, length(tails)÷2)
        dists = collect(Iterators.flatten(dists))
        dims_c = [sum(dims_t)]
        sample = Sample(atomids, nothing, tails, dims_t, dists, nothing, dims_c, y)
        push!(data, sample)
    end
    data
end

function readxyz(filename::String)
    lines = open(readlines, filename)
    buffer = String[]
    mols = Molecule[]
    for line in lines
        push!(buffer, line)
        if isempty(line)
            mol = parsexyz(buffer)
            push!(mols, mol)
            empty!(buffer)
        end
    end
    QM9(mols)
end

function parsexyz(lines::Vector{String})
    natoms = parse(Int, lines[1])
    props = split(lines[2], "\t", keepempty=false)
    T = Float32
    props = [parse(T,props[i]) for i=5:length(props)]
    atoms = Atom[]
    for i = 3:3+natoms-1
        items = split(lines[i], "\t", keepempty=false)
        items = Vector{String}(items)
        symbol = items[1]
        x,y,z = map((2,3,4)) do k
            if occursin("*^", items[k])
                p = split(items[k], "*^")
                v = parse(T,p[1]) ^ parse(Int,p[2])
            else
                v = parse(T,items[k])
            end
            v
        end
        push!(atoms, Atom(symbol,x,y,z))
    end
    m = Molecule(atoms)
    m["props"] = props
    m
end

function readxyzs(dir::String)
    lines = String[]
    for file in readdir(dir)
        path = "$dir/$file"
        println(file)
        @assert endswith(path, ".xyz")
        append!(lines, open(readlines,path))
        push!(lines, "")
    end
    open("qm9.xyz","w") do io
        for line in lines
            println(io, line)
        end
    end
end

function xyz2sdf(filename::String)
    mols = readsdf("$filename.sdf")
    count = 1
    lines = open(readlines, "$filename.xyz")
    buffer = String[]
    for line in lines
        push!(buffer, line)
        if isempty(line)
            n = parse(Int, buffer[1])
            @assert n == natoms(mols[count])
            mols[count]["props"] = buffer[2]
            empty!(buffer)
            count += 1
        end
    end
    writesdf("a.sdf", mols)
end

function QM9_old(mols::Vector{Molecule})
    atom2id = Dict("H"=>1, "C"=>2, "N"=>3, "O"=>4, "F"=>5)
    A = Vector{Int}[]
    B = Matrix{Int}[]
    C = Vector{Float32}[]
    A2B = Vector{Int}[]
    dimsA2B = Vector{Int}[]
    B2A = Matrix{Int}[]
    for m in mols
        push!(A, map(a -> atom2id[a.symbol], m.atoms))
        @assert natoms(m) <= 30

        c = Float32[]
        a2bs = [Int[] for _=1:natoms(m)]
        bondids = Int[]
        for k = 1:nbonds(m)
            b = m.bonds[k]
            a1 = m.atoms[b.i]
            a2 = m.atoms[b.j]
            dist = (a1.x-a2.x)^2 + (a1.y-a2.y)^2 + (a1.z-a2.z)^2
            dist = Float32(1/sqrt(dist))
            push!(c, dist)
            push!(a2bs[b.i], k)
            push!(a2bs[b.j], k)
            aid1 = atom2id[a1.symbol]
            aid2 = atom2id[a2.symbol]
            aid1 = min(aid1, aid2)
            aid2 = max(aid1, aid2)
            bondid = 5(aid1-1) + aid2
            push!(bondids, bondid, natoms(m)+25)
            # push!(bondids, bondid)
        end
        bondids = reshape(bondids, 2, length(bondids)÷2)
        push!(B, bondids)
        push!(C, c)
        a2b = collect(Iterators.flatten(a2bs))
        push!(A2B, a2b)
        push!(dimsA2B, length.(a2bs))

        b2a = map(b -> [b.i,b.j], m.bonds)
        b2a = collect(Iterators.flatten(b2a))
        b2a = reshape(b2a, 2, nbonds(m))
        push!(B2A, b2a)
    end
    Y = map(m -> m["props"][3], mols)
    QM9(mols, A, B, C, A2B, dimsA2B, B2A, Y)
end
