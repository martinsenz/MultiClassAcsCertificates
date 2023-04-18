using Revise, MultiClassAcsCertificates
using MultiClassAcsCertificates.Util: estimate_dirichlet_distribution
using LossFunctions
using ScikitLearn
using Distributions
using LinearAlgebra
using PyCall
using ForwardDiff
using TikzPictures
using PGFPlots
using DataFrames
using Statistics
using CSV

#df = MultiClassAcsCertificates.acquisition()
#df = CSV.read("res/experiments/acquisition.csv", DataFrame)





# # # # # # # #
# Motivation  #
# # # # # # # #

snames = [ "inverse", "improvement", "redistriction", "proportional" ]
MultiClassAcsCertificates.Plots.acquisition("motivation", snames; ternary=false)
MultiClassAcsCertificates.Plots.acquisition("motivation", snames; ternary=true)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Domaingap Varianten bei Prior A = Erwartungswert = pY_T # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# geringe Unsicherheit
snames = [ "proportional" , "domaingap_1Inf_A_low", "domaingap_22_A_low", "domaingap_Inf1_A_low" ]
MultiClassAcsCertificates.Plots.acquisition("domaingap_A_low", snames; ternary=false)
MultiClassAcsCertificates.Plots.acquisition("domaingap_A_low", snames; ternary=true)

# höhere Unsicherheit
snames = [ "proportional" , "domaingap_1Inf_A_high", "domaingap_22_A_high", "domaingap_Inf1_A_high" ]
MultiClassAcsCertificates.Plots.acquisition("domaingap_A_high", snames; ternary=false)
MultiClassAcsCertificates.Plots.acquisition("domaingap_A_high", snames; ternary=true)

snames = [ "proportional" , "domaingap_22_A_high", "domaingap_22_A_low" ]
# MultiClassAcsCertificates.Plots.acquisition("domaingap_A", snames; ternary=false)
MultiClassAcsCertificates.Plots.acquisition("domaingap_A", snames; ternary=true)


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Selektiere für weitere Untersuchungen p=q=2 heraus  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Setze Class Label Prior auf B oder C 
snames = [ "proportional" , "proportional_estimate_B", "domaingap_22_B_high", "domaingap_22_B_low" ]
MultiClassAcsCertificates.Plots.acquisition("domaingap_22_B", snames; ternary=false)
MultiClassAcsCertificates.Plots.acquisition("domaingap_22_B", snames; ternary=true)

snames = [ "proportional" , "proportional_estimate_C", "domaingap_22_C_high", "domaingap_22_C_low" ]
MultiClassAcsCertificates.Plots.acquisition("domaingap_22_C", snames; ternary=false)
MultiClassAcsCertificates.Plots.acquisition("domaingap_22_C", snames; ternary=true)

# # Empirische Varianten?
# snames = [ "proportional" , "proportional_estimate_B", "domaingap_22_B_low", "domaingap_22_empirical_B_low" ]
# MultiClassAcsCertificates.Plots.acquisition("domaingap_22_empirical_B_low", snames; ternary=false)
# MultiClassAcsCertificates.Plots.acquisition("domaingap_22_empirical_B_low", snames; ternary=true)

# snames = [ "proportional" , "proportional_estimate_B", "domaingap_22_B_high", "domaingap_22_empirical_B_high" ]
# MultiClassAcsCertificates.Plots.acquisition("domaingap_22_empirical_B_high", snames; ternary=false)
# MultiClassAcsCertificates.Plots.acquisition("domaingap_22_empirical_B_high", snames; ternary=true)

# snames = [ "proportional" , "proportional_estimate_C", "domaingap_22_C_low", "domaingap_22_empirical_C_low" ]
# MultiClassAcsCertificates.Plots.acquisition("domaingap_22_empirical_C_low", snames; ternary=false)
# MultiClassAcsCertificates.Plots.acquisition("domaingap_22_empirical_C_low", snames; ternary=true)

# snames = [ "proportional" , "proportional_estimate_C", "domaingap_22_C_high", "domaingap_22_empirical_C_high" ]
# MultiClassAcsCertificates.Plots.acquisition("domaingap_22_empirical_C_high", snames; ternary=false)
# MultiClassAcsCertificates.Plots.acquisition("domaingap_22_empirical_C_high", snames; ternary=true)




# # # # # # # # # # # # # # # # # # # # # 
# # ternary Plots for class proportions #
# # # # # # # # # # # # # # # # # # # # # 

# domaingap_A_low
snames = ["domaingap_Inf1_A_low", "domaingap_22_A_low", "domaingap_1Inf_A_low", "proportional"]
df_agg = MultiClassAcsCertificates.Plots.acquisition("domaingap_A_low", snames; ternary=true, color=["tu01", "tu02", "tu03", "tu04"])

# domaingap_A_high
snames = ["domaingap_Inf1_A_high", "domaingap_22_A_high", "domaingap_1Inf_A_high", "proportional"]
df_agg = MultiClassAcsCertificates.Plots.acquisition("domaingap_A_high", snames; ternary=true, color=["tu01", "tu02", "tu03", "tu04"])

snames = ["domaingap_22_B_low", "domaingap_22_B_high", "proportional_estimate_B", "proportional"]
df_agg = MultiClassAcsCertificates.Plots.acquisition("domaingap_22_B", snames; ternary=true, color=["tu01", "tu02", "tu03", "tu04"])

snames = ["domaingap_22_C_low", "domaingap_22_C_high", "proportional_estimate_C", "proportional"]
df_agg = MultiClassAcsCertificates.Plots.acquisition("domaingap_22_C", snames; ternary=true, color=["tu01", "tu02", "tu03", "tu04"])