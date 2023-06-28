module Data
using DataFrames
using Discretizers
using Distributions
using LinearAlgebra
using MetaConfigurations: parsefile
using Random
using StatsBase
using Distances
using PyCall
using AcsCertificates.Data: retrieve

fetch_openml(args...; kwargs...) = pyimport("sklearn.datasets").fetch_openml(args...; kwargs...)
MinMaxScaler(args...; kwargs...) = pyimport("sklearn.preprocessing").MinMaxScaler(args...; kwargs...)

mutable struct DataSet
    X_data::Matrix{Float32}
    y_data::Vector{Int64}
    discretizer::CategoricalDiscretizer
end

"""
    OpenmlDataSet(name; config)

    Retrieve an OpenML `DataSet` by its `name`.
"""
function OpenmlDataSet(name::String; config::Dict{String,Any} = parsefile("conf/data/openml.yml")[name])
    # load openml data based on configurations in conf/data/openml.yml
    bunch = fetch_openml(data_id = config["id"]; as_frame = false)

    # extract and preprocess labels 
    label = [string(l) for l in bunch["target"]]
    y = Vector(encode(CategoricalDiscretizer(label), label))
    if haskey(config, "classes")
        label = _merge_classes(y, config["classes"])
    end
    catdisc = CategoricalDiscretizer(label)
    y = Vector(encode(catdisc, label))

    # extract and preprocess features
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(bunch["data"])

    return DataSet(X, y, catdisc)
end

binary_imbalance = ["fact_balanced", "fact_imbalanced", "protein_homo", "pen_digits", "letter_img", "satimage", "coil_2000", "optical_digits"]

function _merge_classes(y, lb_classes)
    for (i, _) in enumerate(y)
        for (merged_label, classes) in enumerate(lb_classes)
            if y[i] ∈ classes
                y[i] = merged_label
            end
        end
    end
    y
end

"""
    dataset(name)

Return the DataSet object with the given `name`.
"""
function dataset(name::String; kwargs...)
    if name in keys(parsefile("conf/data/openml.yml"))
        return OpenmlDataSet(name; kwargs...)
    elseif name ∈ binary_imbalance
        X,y = retrieve(name)
        catdisc = CategoricalDiscretizer(y)
        return DataSet(X,Vector(encode(catdisc, y)),catdisc)
    else
        throw(KeyError(name))
    end
end

"""
    class_proportion(y, classes)
Returns the ratio of `classes` according to `y` labels. 
"""
function class_proportion(y, classes)
    proportion = StatsBase.proportionmap(y)
    class_proportion = []
    for class in classes
        if haskey(proportion, class)
            push!(class_proportion, proportion[class])
        else
            push!(class_proportion, 0.0)
        end
    end
    convert(Vector{Float64}, class_proportion)
end

"""
    class_counts(y, classes)

Returns the number of class instances according to `y` labels. 
"""
function class_counts(y, classes)
    counts_class = StatsBase.countmap(y)
    counts = []
    for class in classes
        if haskey(counts_class, class)
            push!(counts, counts_class[class])
        else
            @info "There exist no class $(class)"
            push!(counts, 0)
        end
    end
    counts
end

# not used
function sample_indices(y, pY, classes; num_samples=length(y), seed::Integer = convert(Int, rand(UInt32)))
    weighted_pY = pY ./ class_proportion(y, classes)
    StatsBase.sample(MersenneTwister(seed), 1:length(y), Weights(weighted_pY[y]), num_samples, replace=true)
end

function mle_dirichlet(mean_vector, variance; n_samples=1000, margin=0.01)
    if length(mean_vector) == 3
        pop!(mean_vector)
    end
    normal_distr = Distributions.MvNormal(mean_vector, Matrix(variance*I, 2,2))
    X = transpose(rand(normal_distr, n_samples))
    filter_idx = (X[:,1] .>= margin) .& (X[:,2] .>= margin)
    X = X[filter_idx,:]
    diff_vec = fill(1.0, size(X,1)) .- X[:,1] .- X[:,2]
    diff_id = diff_vec .>= margin
    w = max.(0.0, map(x -> Distributions.logpdf(normal_distr, x), eachrow(X[diff_id,:])))
    X = hcat(X[diff_id,:], diff_vec[diff_id,:])
    @info "Sampled points mean $(mean(X, dims=1))"
    @info "Sampled points variance $(var(X, dims=1))"

    d = fit_mle(Distributions.Dirichlet, transpose(X), w)
    @info "Dirichlet alpha: $(d.alpha)"
    @info "Dirichlet mean: $(mean(d))"
    @info "Dirichlet var: $(var(d))"
    d.alpha
end

# generates random drawn class proportions 
function dirichlet_pY(n_samples; α=[0.5,0.5,0.5], margin=0.05, seed::Integer=convert(Int, rand(UInt32)))
    P = transpose(rand(MersenneTwister(seed), Dirichlet(α), n_samples))
    classes = length(α)
    center = fill(1 ./ classes, classes)
    for i in 1:n_samples
       p = P[i,:]
       direction = center .- p      
       P[i,:] = p .+ margin / norm(direction) .* direction
    end
    P
end

_relabeling(y, labels) = replace(y_i -> y_i == y ? 1 : -1, labels)

_binary_labels(x) = map(y -> y==1 ? 1 : -1, x)

function _m_y_from_proportion(pY, m::Int64)
    m_y = round.(Int, pY .* m)
    m_diff = m - sum(m_y) 
    if m_diff > 0
        m_y[argmin(m_y)] += m_diff
    end
    if m_diff < 0
        m_y[argmax(m_y)] += m_diff
    end
    m_y
end

function subsample_indices(y::Vector{Int64}, m_y::Vector{Int64}, classes; seed::Integer=convert(Int, rand(UInt32)))
    y_bool = []
    for class in classes
        push!(y_bool, y .== class)
    end
    min_value = min((class_counts(y, classes) .- m_y)...)
    if min_value < 0.0
        for i in 1:sum(m_y)
            m_y = _m_y_from_proportion(m_y ./ sum(m_y), sum(m_y) - i)
            if min((class_counts(y, classes) .- m_y)...) >= 0.0
                break
            end
        end
    end
    seeds = rand(MersenneTwister(seed), UInt32, length(classes))
    y_indices = []
    for (i,seed) in enumerate(seeds)
        y_indices = vcat(y_indices, StatsBase.sample(MersenneTwister(seed), collect(1:length(y))[y_bool[i]], m_y[i], replace=false))
    end
    y_indices
end

# basic dataset interface
X_data(d::DataSet) = d.X_data
y_data(d::DataSet) = d.y_data
discretizer(d::DataSet) = d.discretizer
num_classes(d::DataSet) = length(keys(discretizer(d).d2n))
classes(d::DataSet) = sort(collect(keys(discretizer(d).d2n))) 

end