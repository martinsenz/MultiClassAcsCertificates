module Experiments 
using
    ..MultiClassAcsCertificates,
    ..MultiClassAcsCertificates.Certification,
    Distributions,
    MetaConfigurations,
    Random,
    ScikitLearn,
    StatsBase,
    PyCall,
    LossFunctions,
    LinearAlgebra,
    DataFrames,
    Distributed,
    CSV,
    PyCall

# ignore all convergence warnings
warnings = pyimport("warnings")
warnings.filterwarnings("ignore") 

function run(configfile::String)
    c = parsefile(configfile)
    funname = "Experiments." * c["job"]    
    @info "Calling $funname(\"$configfile\")"
    fun = eval(Meta.parse(funname))
    fun(configfile) # function call
end

include("exp/tightness.jl")
include("exp/certify.jl")

end