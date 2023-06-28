module Plots
using ..MultiClassAcsCertificates
using CSV, DataFrames, Statistics, PyCall, Distances, StatsBase
using LaTeXStrings
using TikzPictures
using PGFPlots
using Random
using Distributions
using CriticalDifferenceDiagrams

function __init__()
    py"""
    import numpy as np
    import matplotlib.pyplot as plt
    import ternary
    import math
    import itertools

    def contours_coordinates(gap_func, level, scale):
        x_range = np.arange(0, 1.01, 0.01)
        coordinate_list = np.asarray(list(itertools.product(x_range, repeat=2)))
        coordinate_list = np.append(coordinate_list, (1 - coordinate_list[:, 0] - coordinate_list[:, 1]).reshape(-1, 1), axis=1)

        data_list = []
        for point in coordinate_list:
            data_list.append(gap_func(point))
        data_list = np.asarray(data_list)
        data_list[np.sum(coordinate_list[:, 0:2], axis=1) > 1] = np.nan  # remove data outside triangle
        
        # === reshape coordinates and data for use with pyplot contour function
        x = coordinate_list[:, 0].reshape(x_range.shape[0], -1)
        y = coordinate_list[:, 1].reshape(x_range.shape[0], -1)
        h = data_list.reshape(x_range.shape[0], -1)

        # === use pyplot to calculate contours
        contours = plt.contour(x, y, h, level, colors="black", linestyle="dashed")  # this needs to be BEFORE figure definition
        #contours = plt.contour(x, y, h, level)
        plt.clf()  # makes sure that contours are not plotted in carthesian plot

        fig, tax = ternary.figure(scale=scale)

        # === plot contour lines
        for ii, contour in enumerate(contours.allsegs):
            for jj, seg in enumerate(contour):
                tax.plot(seg[:, 0:2] * scale, c="black", linestyle="dashed")

        return fig, tax
    """
end
contours_coordinates(gap_func, level, scale) = py"contours_coordinates"(gap_func, level, scale)
ternary = pyimport("ternary")
simplex_iterator = pyimport("ternary.helpers").simplex_iterator
NearestNDInterpolator= pyimport("scipy.interpolate").NearestNDInterpolator

_to_point(p::Vector{Float64}) = tuple(round.(Int64, p .* 100)...)

function _write_header(io)
    println(io, "\\documentclass[]{article}")
    println(io, "\\usepackage[T1]{fontenc}")
    println(io, "\\usepackage{booktabs}")
    println(io, "\\usepackage{graphicx}")
    println(io, "\\usepackage{amssymb}")
    println(io, "\\usepackage{amsmath}")
    println(io, "\\usepackage{float}")
    println(io, "\\usepackage{subcaption}")
    println(io, "\\setlength{\\tabcolsep}{6pt}")
    println(io, "\\usepackage{float}")
end

function _strategy_names(name)
    if name == "NormedCertificate_Inf_1"
        "\$ \\lVert \\mathbf{d} \\rVert_{\\infty} \\cdot \\lVert \\boldsymbol{\\ell}_{h} \\rVert_{1} \$"
    elseif name == "NormedCertificatePlus_Inf_1"
        "\$ \\lVert \\mathbf{d}_{+} \\rVert_{\\infty} \\cdot \\lVert \\boldsymbol{\\ell}_{h} \\rVert_{1} \$"
    elseif name == "NormedCertificate_2_2"
        "\$ \\lVert \\mathbf{d} \\rVert_{2} \\cdot \\lVert \\boldsymbol{\\ell}_{h} \\rVert_{2} \$"
    elseif name == "NormedCertificatePlus_2_2"
        "\$ \\lVert \\mathbf{d}_{+} \\rVert_{2} \\cdot \\lVert \\boldsymbol{\\ell}_{h} \\rVert_{2} \$"
    elseif name == "NormedCertificate_1_Inf"
        "\$ \\lVert \\mathbf{d} \\rVert_{1} \\cdot \\lVert \\boldsymbol{\\ell}_{h} \\rVert_{\\infty} \$"
    elseif name == "NormedCertificatePlus_1_Inf"
        "\$ \\lVert \\mathbf{d}_{+} \\rVert_{1} \\cdot \\lVert \\boldsymbol{\\ell}_{h} \\rVert_{\\infty} \$"
    else
        name
    end
end

function _classifier_names(name)
    if name === "sklearn.neural_network.MLPClassifier"
        "MLPClassifier"
    elseif name === "sklearn.tree.DecisionTreeClassifier"
        "DecisionTreeClassifier"
    elseif name === "sklearn.linear_model.LogisticRegression"
        "LogisticRegression"
    else
        name
    end
end

# tu-dortmund colors
tu_colors = Dict("tu01" => (0.52, 0.72, 0.09),  # green
                 "tu02" => (0.82, 0.55, 0.07),  # brown
                 "tu03" => (0.10, 0.71, 0.71),  # cyan
                 "tu04" => (0.97, 0.35, 0.24),  # red
                 "tu05" => (0.29, 0.42, 0.99))  # blue


include("plt/datasets.jl")
include("plt/tightness.jl")
include("plt/certify.jl")
include("plt/acquisition.jl")

end