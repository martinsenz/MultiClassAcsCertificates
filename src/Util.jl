module Util 
using PyCall

# python imports
compute_sample_weight(args...; kwargs...) = pyimport("sklearn.utils.class_weight").compute_sample_weight(args...; kwargs...)
ternary = pyimport("ternary")
simplex_iterator = pyimport("ternary.helpers").simplex_iterator
NearestNDInterpolator = pyimport("scipy.interpolate").NearestNDInterpolator

end