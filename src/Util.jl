module Util 
using PyCall
using Distributions

# python imports
compute_sample_weight(args...; kwargs...) = pyimport("sklearn.utils.class_weight").compute_sample_weight(args...; kwargs...)

function estimate_dirichlet_distribution(d; only_parameters=false, n_samples=1000000, verbose=true)
    X = max.(0.01, rand(d, n_samples))
    w = map(x -> Distributions.pdf(d,x), eachrow(transpose(X)))
    dirichlet = Distributions.fit_mle(Distributions.Dirichlet, X ./ sum(X, dims=1), w)
    if verbose
        @info "Distribution d: mean(d)=$(mean(d)), var(d)=$(var(d)) \n 
        Dirichlet fitting => mean=$(round.(mean(dirichlet);digits=4)), var=$( round.( var(dirichlet); digits=4 ) )"
    end
    if only_parameters
        dirichlet.alpha
    else
        dirichlet
    end
end

end