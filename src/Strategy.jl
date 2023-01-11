module Strategy
using ..Certification: NormedCertificate_Inf_1, NormedCertificate_2_2, NormedCertificate_1_Inf, _ϵ
using Random
using StatsBase
using Distributions
using ScikitLearn
using FiniteDifferences
using SimplexQuad
using LinearAlgebra
using ForwardDiff


function ℓNorm(c::NormedCertificate_Inf_1, m)
    sum(c.empirical_ℓ_y) + sum(map(i -> _ϵ(m[i], c.δ_y[i]), 1:length(m)))
end

function ℓNorm(c::NormedCertificate_1_Inf, m)
    maximum(c.empirical_ℓ_y .+ map(i -> _ϵ(m[i], c.δ_y[i]), 1:length(m)))
end

function ∇ϵ(c::Union{NormedCertificate_Inf_1, NormedCertificate_1_Inf}, 
            class_prior_distribution::Distributions.MultivariateDistribution,
            ;n_points=10000)

    starting_point = sum(c.m_y) .* c.pY_S
    -ForwardDiff.gradient(m -> ϵ(c, class_prior_distribution, m; n_points=n_points), starting_point)
end

function ϵ(c::Union{NormedCertificate_Inf_1, NormedCertificate_1_Inf}, 
        class_prior_distribution::Distributions.MultivariateDistribution,
        m,
        ;n_points=10000)
 
    _ps(m) = m ./ sum(m)
    _deviation_norm(c::NormedCertificate_Inf_1, x::Vector{Float64}) = norm(_ps(m) .- x, Inf)
    _deviation_norm(c::NormedCertificate_1_Inf, x::Vector{Float64}) = norm(_ps(m) .- x, 1)
    
    # integration of ∫f(m)
    naive_montecarlointegration(
        p -> Distributions.pdf(class_prior_distribution, p) * _deviation_norm(c, p) * ℓNorm(c, m),
        n_points,
        length(c.classes))
end

function ϵ(c::Union{NormedCertificate_Inf_1, NormedCertificate_1_Inf}, 
        class_prior_distribution::Distributions.MultivariateDistribution,
        m::Vector{Int64}; n_points=10000)

    _ps(m) = m ./ sum(m)
    _deviation_norm(c::NormedCertificate_Inf_1, x::Vector{Float64}) = norm(_ps(m) .- x, Inf)
    _deviation_norm(c::NormedCertificate_1_Inf, x::Vector{Float64}) = norm(_ps(m) .- x, 1)
    
    # integration of ∫f(m)
    naive_montecarlointegration(
        p -> Distributions.pdf(class_prior_distribution, p) * _deviation_norm(c, p) * ℓNorm(c, m),
        n_points,
        length(c.classes))
end

function estimate_dirichlet_distribution(d; n_samples=1000, verbose=false)
    X = max.(0.01, rand(d, n_samples))
    w = map(x -> Distributions.pdf(d,x), eachrow(transpose(X)))
    dirichlet = Distributions.fit_mle(Distributions.Dirichlet, X ./ sum(X, dims=1), w)
    if verbose
        @info "Distribution d: mean(d)=$(mean(d)), var(d)=$(var(d)) \n 
        Dirichlet fitting => mean=$(round.(mean(dirichlet);digits=4)), var=$( round.( var(dirichlet); digits=4 ) )"
    end
    dirichlet
end

function naive_montecarlointegration(f::Function, N::Int64, n_classes::Int64; seed=1234)
    random_points = transpose(rand(MersenneTwister(seed), Distributions.Dirichlet(ones(n_classes)), N))
    f_sum = 0.0 
    for i in 1:N
        f_sum += f(random_points[i,:])
    end
    V = 1 / factorial(n_classes-1)
    V * (1 / N) * f_sum 
end

end