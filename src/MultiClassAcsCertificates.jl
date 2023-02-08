__precompile__(false)
module MultiClassAcsCertificates
using PyCall
export Data
export Certification
export Util
export Experiments
export Plots
export SkObject
export NormedCertificate, SignedCertificate, DomainGap


# function __init__()
#     Conda.add_channel("conda-forge", priority=:never)
# end



"""
    SkObject(class_name, configuration)
    SkObject(class_name; kwargs...)

Instantiate a scikit-learn `PyObject` by its fully qualified `class_name`.
"""
SkObject(class_name::AbstractString, config) =
    SkObject(class_name; [ Symbol(k) => v for (k, v) in config ]...)
SkObject(class_name::AbstractString; kwargs...) =
    getproperty(
        pyimport(join(split(class_name, ".")[1:end-1], ".")), # package
        Symbol(split(class_name, ".")[end]) # classname
    )(; kwargs...) # constructor call: package.classname(**kwargs)

include("Data.jl")
using .Data

include("Util.jl")
using .Util

include("Plots.jl")
using .Plots

include("Certification.jl")
using .Certification

include("Strategy.jl")
using .Strategy

include("Experiments.jl")
using .Experiments

"""
    certify(config_path)
    
Conduct certification experiments for multi-class problems and produce Table 1.
"""
function certify(config_path="conf/exp/certify.yml")
    Experiments.certify(config_path)
    Plots.certify()
end

"""
    tightness(config_path, standalone, full_table)

Conduct tightness exoeriments to produce Table 2 Fig.1 and Fig.2
"""
function tightness(config_path="conf/exp/tightness.yml", standalone=true, full_table=false)
    Experiments.tightness(config_path)
    Plots.tightness(standalone, full_table)
end

function acquisition(config_path="conf/exp/acquisition.yml")
    Experiments.acquisition(config_path)
end

"""
    run_all_experiments()

Conduct all experiments 
"""
function run_all_experiments()
    certify()
    tightness()
end

end