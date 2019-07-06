using LightChem
using JSON
using Merlin

include("data.jl")
include("model.jl")
include("nn.jl")

config = JSON.parsefile(ARGS[1])
Merlin.setdevice(config["device"])

if config["train"]
    # xyz2sdf(".data/gdb9")
    m = Model(config)
    #LightNLP.save("ner.jld2", ner)
else

end
println("Finish.")


function read_gdb9(filename::String)
    throw("Not implemented yet.")
    lines = open(readlines, filename)
    headers = split(lines[1], ",")
    headerdict = Dict(i=>headers[i] for i=1:length(headers))
    qm9_tasks = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0", "u298", "h298", "g298"]
    v = Float32[]
    for i = 2:length(lines)
        items = split(lines[i], ",")
        molid = items[1]
        values = map(x -> parse(Float32,x), items[2:end])
        append!(v, values)
    end
    n = length(lines) - 1
    v = reshape(v, length(v)÷n, n)
    v
end
