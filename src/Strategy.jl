module Strategy
using ..Certification: NormedCertificate_Inf_1, NormedCertificate_2_2, NormedCertificate_1_Inf, _ϵ, NormedCertification
using Random
using StatsBase
using Distributions
using ScikitLearn
using LinearAlgebra
using ForwardDiff

function ℓNorm(c::NormedCertificate_Inf_1, m)
    if c.pac_bounds
        sum(c.empirical_ℓ_y) + sum(map(i -> _ϵ(m[i], c.δ_y[i]), 1:length(m)))
    else
        norm(c.empirical_ℓ_y, 1)
    end
end

function ℓNorm(c::NormedCertificate_2_2, m)
    if c.pac_bounds
        sqrt(sum(map(i -> (c.empirical_ℓ_y[i] + _ϵ(m[i], c.δ_y[i]))^2, 1:length(m))))
    else
        norm(c.empirical_ℓ_y, 2)
    end
end

function ℓNorm(c::NormedCertificate_1_Inf, m)
    if c.pac_bounds
        maximum(c.empirical_ℓ_y .+ map(i -> _ϵ(m[i], c.δ_y[i]), 1:length(m)))
    else
        norm(c.empirical_ℓ_y, Inf)
    end
end

function ∇ϵ(c::NormedCertification, 
            class_prior_distribution::Distributions.MultivariateDistribution,
            ;n_points=10000, plus=false, seed=123)

    starting_point = sum(c.m_y) .* c.pY_S
    -ForwardDiff.gradient(m -> ϵ(c, class_prior_distribution, m; n_points=n_points, plus=plus, seed=seed), starting_point)
end

function ϵ(c::NormedCertification, 
        class_prior_distribution::Distributions.MultivariateDistribution,
        m,
        ;n_points=10000, plus=false, seed=123)
 
    _ps(m) = m ./ sum(m)
    _deviation_norm(c::NormedCertificate_Inf_1, x::Vector{Float64}) = norm(plus ? max.(0.0, _ps(m) .- x) : _ps(m) .- x, Inf)
    _deviation_norm(c::NormedCertificate_2_2, x::Vector{Float64}) = norm(plus ? max.(_ps(m) .- x) : _ps(m) .- x, 2)
    _deviation_norm(c::NormedCertificate_1_Inf, x::Vector{Float64}) = norm(plus ? max.(0.0, _ps(m) .- x) : _ps(m) .- x, 1)
    
    # integration of ∫f(m)
    naive_montecarlointegration(
        p -> Distributions.pdf(class_prior_distribution, p) * _deviation_norm(c, p) * ℓNorm(c, m),
        n_points,
        length(c.classes);
        seed=seed)
end

function ϵ(c::NormedCertification, 
        class_prior_distribution::Distributions.MultivariateDistribution,
        m::Vector{Int64}; plus=false, n_points=10000)

    _ps(m) = m ./ sum(m)
    _deviation_norm(c::NormedCertificate_Inf_1, x::Vector{Float64}) = norm(plus ? max.(0.0, _ps(m) .- x) : _ps(m) .- x, Inf)
    _deviation_norm(c::NormedCertificate_2_2, x::Vector{Float64}) = norm(plus ? max.(_ps(m) .- x) : _ps(m) .- x, 2)
    _deviation_norm(c::NormedCertificate_1_Inf, x::Vector{Float64}) = norm(plus ? max.(0.0, _ps(m) .- x) : _ps(m) .- x, 1)
    
    # integration of ∫f(m)
    naive_montecarlointegration(
        p -> Distributions.pdf(class_prior_distribution, p) * _deviation_norm(c, p) * ℓNorm(c, m),
        n_points,
        length(c.classes))
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

function suggest_acquisition(c::NormedCertification,
                            class_prior_distribution::Distributions.MultivariateDistribution, 
                            batchsize::Int64;
                            n_samples_mc::Int64=10000, plus=false, threshold=0.000000001, seed=123, warn=false)

    gradient = ∇ϵ(c, class_prior_distribution; n_points=n_samples_mc, plus=plus, seed=seed)
    if all(g -> g <= threshold, gradient) 
        if warn
            @warn "Determined gradient $(gradient) is negative!"
        end
        gradient = mean(class_prior_distribution) .* (sum(c.m_y) + batchsize)
    end
    acquisition = round.(Int64, max.(0.0, gradient) .* batchsize / sum(gradient[gradient.>0]))
    d = abs(sum(acquisition) - batchsize)
    sample_idx_diff = StatsBase.sample(c.classes, StatsBase.Weights(mean(class_prior_distribution)), d)
    if d < 0.0
        acquisition .+= values(sort(countmap(sample_idx_diff)))
    elseif d > 0.0
        acquisition .-= values(sort(countmap(sample_idx_diff)))
    end
    acquisition
end

end