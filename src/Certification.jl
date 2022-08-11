module Certification
using AcsCertificates
using LossFunctions
using ..Data
using ..Util
using LinearAlgebra
using JuMP
using NLopt
export domaingap_error, empirical_classwise_risk, MultiClassCertificate, p_range, _ϵ

# one-sided maximum error with probability at least 1-δ
_ϵ(m::Int, δ::Float64) = sqrt(-log(δ) / (2*m))
_δ(m, ϵ) = exp(-2 * m * ϵ^2)

"""
    empirical_classwise_risk(L, y_h, y)
The class-wise risk of predictions `y_h` under the loss function `L`.
"""
function empirical_classwise_risk(L, y_h, y, classes) 
    classwise_risk = []  
    for y_i in classes
        y_binary = Data._relabeling(y_i, y[y.==y_i])
        ŷ_binary = Data._relabeling(y_i, y_h[y.==y_i])
        push!(classwise_risk, LossFunctions.value(L, y_binary, ŷ_binary, AggMode.Mean()))
    end
    classwise_risk
end

"""
    MultiClassCertificate(L, y_h, y)

A multiclass Certificate about the robustness of an hypothesis `h` with respect to changes in the class proportions. 
This certificate based on the hoelder estimate |d_{+}|_{∞} * |l|_{1}. 

You can inspect this certificate using the *Methods* listed below. It is based
on the predictions `y_h` and ground-truth class labels `y ∈ {1, ..., N}` and holds
for the loss function `L` with probability at least `1 - δ`.

### Keyword Arguments
- `δ = 0.05`
- `classes = sort(unique(y))` class labels
- `w_y = [1.,...,1.]` optional class weights
- `pac_bounds = true` used upper bounded classwise loss
- `tol = 1e-4` the tolerance, `tol > 0` for the constrained optimization
- `n_trials = 3` number of trials in the multi-start global optimization
"""
struct MultiClassCertificate
    ℓNormBounded :: Float64
    ℓNorm :: Float64
    δ_y :: Vector{Float64}
    empirical_ℓ_y ::Vector{Float64}
    m_y :: Vector{Int}
    L::SupervisedLoss
    δ::Float64
    classes::Vector{Int64}
    w_y::Vector{Float64}
    pac_bounds::Bool
    function MultiClassCertificate(L, y_h, y; δ=0.05, classes=sort(unique(y)), w_y=fill(1., length(unique(y))), pac_bounds=true, tol=1e-4, n_trials=3)
        m_y = Data.class_counts(y, classes)
        empirical_ℓ_y = w_y .* empirical_classwise_risk(L, y_h, y, classes)
        ℓ_norm = norm(empirical_ℓ_y, 1)
        ϵ_bound, δ_y = begin  
            if pac_bounds
                optimize_error(m_y, δ; tol=tol, n_trials=n_trials)
            else
                0.0, fill(0.0, length(classes))
            end
        end
        new(ℓ_norm + ϵ_bound, ℓ_norm, δ_y, empirical_ℓ_y, m_y, L, δ, classes, w_y, pac_bounds)
    end  
end

function _optimize_error(m_y, δ; tol=1e-4)
    N = length(m_y)
    model = Model(NLopt.Optimizer)
    register(model, :_δ, 2, _δ; autodiff = true)
    set_optimizer_attribute(model, "algorithm", :GN_ISRES) # :LD_MMA, :LD_SLSQP
    set_optimizer_attribute(model, "maxtime", 5.0)
    @variable(model, tol <= ϵ[1:N] <= 1/tol)
    @NLconstraint(model, sum(_δ(m_y[i], ϵ[i]) for i=1:N) <= δ)
    @NLobjective(model, Min, sum(ϵ[i] for i=1:N))
    JuMP.optimize!(model)
    ϵ_y = vcat((JuMP.value(ϵ[i]) for i in 1:N)...)
    δ_y = vcat((_δ(m_y[i], ϵ_y[i]) for i in 1:N)...)
    sum(ϵ_y), δ_y
end

function optimize_error(m_y, δ; tol=1e-4, n_trials=3)
    min_ϵ = Inf
    min_δ_y = fill(0.0, length(m_y))
    for i in 1:n_trials
        ϵ, δ_y = _optimize_error(m_y, δ; tol=tol)
        if ϵ < min_ϵ
            min_ϵ = ϵ
            min_δ_y = δ_y
        end
    end
    min_ϵ, min_δ_y
end

"""
    domaingap_error(c, Py_S, Py_T)

Prediction of the ACS-induced error by a given label shift.
"""
function domaingap_error(c::MultiClassCertificate, Py_S, Py_T; pac_bounds=true, variant_plus=false)
    d = Py_T .- Py_S
    if variant_plus
        d = max.(d, 0.0)
    end
    d = norm(d, Inf)
    if pac_bounds
        return d * c.ℓNormBounded
    else
        return d * c.ℓNorm
    end
end

function domaingap_error(method_name, ℓNormBounded, Py_S, Py_T; variant_plus=false)
    d = Py_T .- Py_S
    if variant_plus
        d = max.(d, 0.0)
    end
    if method_name in [ "HoelderCertificateInf1", "HoelderCertificatePlusInf1" ]
        return norm(d, Inf) .* ℓNormBounded
    else
        @error "method name $(method_name) not recognized!"
    end
end

"""
    max_Δp(c, ϵ)

The maximum classwise absolute deviation Δp for which the hypothesis certified by `c` 
has an ACS-induced error of less then `ϵ`.
"""
function max_Δp(c::MultiClassCertificate, ϵ::Float64)
    max_p = ϵ / c.ℓNormBounded
    max_p
end

"""
    p_range(c, ϵ)

The maximum range of values for `p_T` for which the hypothesis certified by `c` has an ACS-induced error of less than `ϵ`.
"""
function p_range(c::MultiClassCertificate, ϵ::Float64)
    p_S = c.m_y ./ sum(c.m_y)
    vcat(([[p_S[i] - max_Δp(c, ϵ), p_S[i] + max_Δp(c, ϵ)]] for i in 1:length(c.classes))...)
end

function Base.show(io::IO, ::MIME"text/plain", c::MultiClassCertificate)
    println(io, "┌ Certificate on label shift robustness:")
    println(io, "│ * max_Δp = ", max_Δp(c, 0.05), " at ϵ=0.05")
    println(io, "│ * p_range at ϵ=0.05:")
    for i in 1:length(c.m_y)
        println(io, "│ class $(i) ", p_range(c, 0.05)[i])
    end
    println("│ - - - - - - - - - - - - - - - - - - - - - -")
    println(io, "│ * max_Δp = ", max_Δp(c, 0.1), " at ϵ=0.1")
    println(io, "│ * p_range at ϵ=0.1:")
    for i in 1:length(c.m_y)
        println(io, "│ class $(i) ", p_range(c, 0.1)[i])
    end
    println("│ - - - - - - - - - - - - - - - - - - - - - -")
    println(io, "└─┬ Given the following scenario:")
    println(io, "  │ * effective δ = $(round(sum(c.δ_y);digits=4))")
    println(io, "  │ * δ_y = $(round.(c.δ_y;digits=6))")
    println(io, "  │ * empirical_ℓ_y = ",
        round.(c.empirical_ℓ_y; digits=6))
    println(io, "  │ * w_y = ", round.(c.w_y; digits=6))
    println(io, "  │ * m_y = ", c.m_y)
    println(io, "  └ * L = ", c.L)
end 




end