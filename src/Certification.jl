module Certification
using LossFunctions
using ..Data
using LinearAlgebra
using JuMP
using NLopt
export domaingap_error, empirical_classwise_risk, NormedCertificate, p_range, _ϵ

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

# supertype of the normed certifications
abstract type NormedCertification end

"""
    NormedCertificate(L, y_h, y)

A multiclass certificate about the robustness of an hypothesis `h` with respect to changes in the class proportions. 
This certificate based on the hoelder estimate |d|_{p} * |l|_{q} with conjugates p an q. 

You can inspect this certificate using the *Methods* listed below. It is based
on the predictions `y_h` and ground-truth class labels `y ∈ {1, ..., N}` and holds
for the loss function `L` with probability at least `1 - δ`.

### Keyword Arguments
- `hoelder_conjugate = Inf_1` selectable conjugates with `Inf_1` (∞,1), `2_2` (2,2) and `1_Inf` (1,∞)
- `δ = 0.05` probabilty budget of `minimize_ℓ_norm_error` 
- `classes = sort(unique(y))` class labels
- `w_y = [1.,...,1.]` optional class weights
- `pac_bounds = true` used upper bounded classwise loss
- `tol = 1e-4` the tolerance, `tol > 0` for the constrained optimization
- `n_trials = 3` number of trials in the multi-start global optimization
"""
function NormedCertificate(L, y_h, y; kwargs...)
    kwargs = Dict(kwargs)
    hoelder_conjugate = get(kwargs, :hoelder_conjugate, "Inf_1")
    configuration = Dict(
        :δ => get(kwargs, :δ, 0.05),
        :classes => get(kwargs, "classes", sort(unique(y))), 
        :w_y => get(kwargs, :w_y, fill(1., length(unique(y)))),
        :pac_bounds => get(kwargs, :pac_bounds, true),
        :tol => get(kwargs, :tol, 1e-4),
        :n_trials => get(kwargs, :n_trials, 3)
    )
    if hoelder_conjugate == "Inf_1"
        NormedCertificate_Inf_1(L, y_h, y; configuration...)
    elseif hoelder_conjugate == "2_2"
        NormedCertificate_2_2(L, y_h, y; configuration...)
    elseif hoelder_conjugate == "1_Inf"
        NormedCertificate_1_Inf(L, y_h, y; configuration...)
    else
        throw(ArgumentError("`hoelder_conjugate` not recognized!"))
    end
end

"""
    NormedCertificate_1_Inf(L, y_h, y) <: NormedCertification

This certificate based on the hoelder estimate |d|_{1} * |l|_{∞}.
See `NormedCertificate` for documentation.
"""
struct NormedCertificate_1_Inf <: NormedCertification
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
    function NormedCertificate_1_Inf(L, y_h, y; δ=0.05, classes=sort(unique(y)), w_y=fill(1., length(unique(y))), pac_bounds=true, tol=1e-4, n_trials=3)
        m_y = Data.class_counts(y, classes)
        empirical_ℓ_y = w_y .* empirical_classwise_risk(L, y_h, y, classes)
        ℓ_norm = norm(empirical_ℓ_y, Inf) 
        ϵ_bound, δ_y = begin  
            if pac_bounds
                minimize_ℓ_norm_error("Inf_1", empirical_ℓ_y, m_y, δ; tol=tol, n_trials=n_trials)
            else
                0.0, fill(0.0, length(classes))
            end
        end
        new(ℓ_norm + ϵ_bound, ℓ_norm, δ_y, empirical_ℓ_y, m_y, L, δ, classes, w_y, pac_bounds)
    end  
end

"""
    NormedCertificate_2_2(L, y_h, y) <: NormedCertification

This certificate based on the hoelder estimate |d|_{2} * |l|_{2}.
See `NormedCertificate` for documentation.
"""
struct NormedCertificate_2_2 <: NormedCertification
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
    function NormedCertificate_2_2(L, y_h, y; δ=0.05, classes=sort(unique(y)), w_y=fill(1., length(unique(y))), pac_bounds=true, tol=1e-4, n_trials=3)
        m_y = Data.class_counts(y, classes)
        empirical_ℓ_y = w_y .* empirical_classwise_risk(L, y_h, y, classes)
        ℓ_norm = norm(empirical_ℓ_y, 2) 
        ϵ_bound, δ_y = begin  
            if pac_bounds
                minimize_ℓ_norm_error("2_2", empirical_ℓ_y, m_y, δ; tol=tol, n_trials=n_trials)
            else
                0.0, fill(0.0, length(classes))
            end
        end
        new(ℓ_norm + ϵ_bound, ℓ_norm, δ_y, empirical_ℓ_y, m_y, L, δ, classes, w_y, pac_bounds)
    end  
end

"""
    NormedCertificate_Inf_1(L, y_h, y) <: NormedCertification

This certificate based on the hoelder estimate |d|_{∞} * |l|_{1}.
See `NormedCertificate` for documentation.
"""
struct NormedCertificate_Inf_1 <: NormedCertification
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
    function NormedCertificate_Inf_1(L, y_h, y; δ=0.05, classes=sort(unique(y)), w_y=fill(1., length(unique(y))), pac_bounds=true, tol=1e-4, n_trials=3)
        m_y = Data.class_counts(y, classes)
        empirical_ℓ_y = w_y .* empirical_classwise_risk(L, y_h, y, classes)
        ℓ_norm = norm(empirical_ℓ_y, 1) 
        ϵ_bound, δ_y = begin  
            if pac_bounds
                minimize_ℓ_norm_error("Inf_1", empirical_ℓ_y, m_y, δ; tol=tol, n_trials=n_trials)
            else
                0.0, fill(0.0, length(classes))
            end
        end
        new(ℓ_norm + ϵ_bound, ℓ_norm, δ_y, empirical_ℓ_y, m_y, L, δ, classes, w_y, pac_bounds)
    end  
end

function minimize_ℓ_norm_error(hoelder_conjugate, empirical_ℓ_y, m_y, δ; tol=1e-4, n_trials=3)
    min_ϵ = Inf
    min_δ_y = fill(0.0, length(m_y))
    for _ in 1:n_trials
        ϵ, δ_y = _minimize_ℓ_norm_error(hoelder_conjugate, empirical_ℓ_y, m_y, δ; tol=tol)
        if ϵ < min_ϵ
            min_ϵ = ϵ
            min_δ_y = δ_y
        end
    end
    min_ϵ, min_δ_y
end

function _minimize_ℓ_norm_error(hoelder_conjugate, empirical_ℓ_y, m_y, δ; tol=1e-4)
    N = length(m_y)
    model = Model(NLopt.Optimizer)
    register(model, :_δ, 2, _δ; autodiff = true)
    set_optimizer_attribute(model, "algorithm", :GN_ISRES) # :LD_MMA, :LD_SLSQP
    set_optimizer_attribute(model, "maxtime", 5.0)
    @variable(model, tol <= ϵ[1:N] <= 1/tol)
    @NLconstraint(model, sum(_δ(m_y[i], ϵ[i]) for i=1:N) <= δ)
    if hoelder_conjugate == "Inf_1" # optimize l_1 norm
        @NLobjective(model, Min, sum(ϵ[i] for i=1:N))
        JuMP.optimize!(model)
        ϵ_y = vcat((JuMP.value(ϵ[i]) for i in 1:N)...)
        δ_y = vcat((_δ(m_y[i], ϵ_y[i]) for i in 1:N)...)
        sum(ϵ_y), δ_y
    elseif hoelder_conjugate == "2_2" # optimize l2 norm
        @NLobjective(model, Min, sum((ϵ[i] + empirical_ℓ_y[i])^2 for i=1:N))
        JuMP.optimize!(model)
        ϵ_y = vcat((JuMP.value(ϵ[i]) for i in 1:N)...)
        δ_y = vcat((_δ(m_y[i], ϵ_y[i]) for i in 1:N)...)
        sqrt(sum(_δ(m_y[i], ϵ_y[i]) for i=1:N)), δ_y
    elseif hoelder_conjugate == "1_Inf" # optimize l_∞ norm
        sort_id = sortperm(empirical_ℓ_y, rev=true)
        id_max = sort_id[1]
        @NLobjective(model, Min, ϵ[id_max])
        for i in 1:N 
            @NLconstraint(model, (ϵ[id_max]-ϵ[i]) >= (empirical_ℓ_y[sort_id][i]-empirical_ℓ_y[id_max]) )
        end
        JuMP.optimize!(model)
        ϵ_y = vcat((JuMP.value(ϵ[i]) for i in 1:N)...)
        δ_y = vcat((_δ(m_y[i], ϵ_y[i]) for i in 1:N)...)
        norm(empirical_ℓ_y[sort_id] .+ ϵ_y, Inf), δ_y[sort_id]
    else
        @error "Does not recognized hoelder_conjugate = $(hoelder_conjugate)!"
    end
end

"""
    domaingap_error(c, Py_S, Py_T)

Prediction of the ACS-induced error by a given label shift.
"""
function domaingap_error(c::NormedCertificate_1_Inf, Py_S, Py_T; pac_bounds=true, variant_plus=false)
    d = Py_T .- Py_S
    if variant_plus
        d = max.(d, 0.0)
    end
    d = norm(d, 1)
    if pac_bounds
        return d * c.ℓNormBounded
    else
        return d * c.ℓNorm
    end
end

function domaingap_error(c::NormedCertificate_2_2, Py_S, Py_T; pac_bounds=true, variant_plus=false)
    d = Py_T .- Py_S
    if variant_plus
        d = max.(d, 0.0)
    end
    d = norm(d, 2)
    if pac_bounds
        return d * c.ℓNormBounded
    else
        return d * c.ℓNorm
    end
end

function domaingap_error(c::NormedCertificate_Inf_1, Py_S, Py_T; pac_bounds=true, variant_plus=false)
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
    if method_name ∈ [ "NormedCertificate_Inf_1", "NormedCertificatePlus_Inf_1" ]
        norm(d, Inf) .* ℓNormBounded
    elseif method_name ∈ [ "NormedCertificate_2_2", "NormedCertificatePlus_2_2" ]
        norm(d, 2) .* ℓNormBounded
    elseif method_name ∈ [ "NormedCertificate_1_Inf", "NormedCertificatePlus_1_Inf" ]
        norm(d, 1) .* ℓNormBounded
    else
        @error "method name $(method_name) not recognized!"
    end
end

"""
    max_Δp(c, ϵ)

The maximum normwise absolute deviation Δp for which the hypothesis certified by `c` 
has an ACS-induced error of less then `ϵ`.
"""
function max_Δp(c::NormedCertification, ϵ::Float64)
    if c.pac_bounds
        ϵ / c.ℓNormBounded
    else
        ϵ / c.ℓNorm
    end
end

"""
    p_range(c, ϵ)

The maximum range of values for `p_T` for which the hypothesis certified by `c` has an ACS-induced error of less than `ϵ`.
"""
function p_range(c::NormedCertificate_Inf_1, ϵ::Float64)
    p_S = c.m_y ./ sum(c.m_y)
    vcat(([[p_S[i] - max_Δp(c, ϵ), p_S[i] + max_Δp(c, ϵ)]] for i in 1:length(c.classes))...)
end

function p_range(c::NormedCertification, ϵ::Float64)
    throw(ErrorException("`p_range` is only available for the hoelder conjugate (∞,1)"))
end

# some output functions 
_print_hoelder_conjugate(c::NormedCertificate_Inf_1) = "(∞,1)"
_print_hoelder_conjugate(c::NormedCertificate_2_2) = "(2,2)"
_print_hoelder_conjugate(c::NormedCertificate_1_Inf) = "(1,∞)"

function Base.show(io::IO, ::MIME"text/plain", c::NormedCertificate_Inf_1)
    println(io, "┌ Normed certificate $(_print_hoelder_conjugate(c)) on label shift robustness:")
    println(io, "│ * max_Δp = ", max_Δp(c, 0.05), " at ϵ=0.05")
    println(io, "│ * p_range at ϵ=0.05:")
    for i in 1:length(c.m_y)
        println(io, "│ class $(i) ", min.(1.0, max.(0.0, p_range(c, 0.05)[i])))
    end
    println("│ - - - - - - - - - - - - - - - - - - - - - -")
    println(io, "│ * max_Δp = ", max_Δp(c, 0.1), " at ϵ=0.1")
    println(io, "│ * p_range at ϵ=0.1:")
    for i in 1:length(c.m_y)
        println(io, "│ class $(i) ", min.(1.0, max.(0.0, p_range(c, 0.1)[i])))
    end
    println("│ - - - - - - - - - - - - - - - - - - - - - -")
    println(io, "└─┬ Given the following scenario:")
    if c.pac_bounds
        println(io, "  │ * effective δ = $(round(sum(c.δ_y);digits=4))")
        println(io, "  │ * δ_y = $(round.(c.δ_y;digits=6))")
    else
        println(io, "  │ * pac bounds = false")
    end
    println(io, "  │ * empirical_ℓ_y = ",
        round.(c.empirical_ℓ_y; digits=6))
    println(io, "  │ * w_y = ", round.(c.w_y; digits=6))
    println(io, "  │ * m_y = ", c.m_y)
    println(io, "  │ * p_s = ", round.(c.m_y ./ sum(c.m_y); digits=4))
    println(io, "  └ * L = ", c.L)
end 

function Base.show(io::IO, ::MIME"text/plain", c::NormedCertification)
    println(io, "┌ Normed certificate $(_print_hoelder_conjugate(c)) on label shift robustness:")
    println(io, "│ * max_Δp = ", max_Δp(c, 0.05), " at ϵ=0.05")
    println("│ - - - - - - - - - - - - - - - - - - - - - -")
    println(io, "│ * max_Δp = ", max_Δp(c, 0.1), " at ϵ=0.1")
    println("│ - - - - - - - - - - - - - - - - - - - - - -")
    println(io, "└─┬ Given the following scenario:")
    if c.pac_bounds
        println(io, "  │ * effective δ = $(round(sum(c.δ_y);digits=4))")
        println(io, "  │ * δ_y = $(round.(c.δ_y;digits=6))")
    else
        println(io, "  │ * pac bounds = false")
    end
    println(io, "  │ * empirical_ℓ_y = ",
        round.(c.empirical_ℓ_y; digits=6))
    println(io, "  │ * w_y = ", round.(c.w_y; digits=6))
    println(io, "  │ * m_y = ", c.m_y)
    println(io, "  │ * p_s = ", round.(c.m_y ./ sum(c.m_y); digits=4))
    println(io, "  └ * L = ", c.L)
end 

end