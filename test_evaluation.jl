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

snames = [ "random", "inverse", "improvement", "redistriction", "proportional" ]
MultiClassAcsCertificates.Plots.acquisition("motivation", snames; ternary=false)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Domaingap Varianten bei Prior A = Erwartungswert = pY_T # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# geringe Unsicherheit
snames = [ "proportional" , "domaingap_1Inf_A_low", "domaingap_22_A_low", "domaingap_Inf1_A_low" ]
MultiClassAcsCertificates.Plots.acquisition("domaingap_A_low", snames; ternary=false)

# höhere Unsicherheit
snames = [ "proportional" , "domaingap_1Inf_A_high", "domaingap_22_A_high", "domaingap_Inf1_A_high" ]
MultiClassAcsCertificates.Plots.acquisition("domaingap_A_high", snames; ternary=false)


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Selektiere für weitere Untersuchungen p=q=2 heraus  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Setze Class Label Prior auf B oder C 
snames = [ "proportional" , "proportional_estimate_B", "domaingap_22_B_high", "domaingap_22_B_low" ]
MultiClassAcsCertificates.Plots.acquisition("domaingap_22_B", snames; ternary=false)

snames = [ "proportional" , "proportional_estimate_C", "domaingap_22_C_high", "domaingap_22_C_low" ]
MultiClassAcsCertificates.Plots.acquisition("domaingap_22_C", snames; ternary=true)

# Empirische Varianten?
snames = [ "proportional" , "proportional_estimate_B", "domaingap_22_B_low", "domaingap_22_empirical_B_low" ]
MultiClassAcsCertificates.Plots.acquisition("domaingap_22_empirical_B_low", snames; ternary=false)

snames = [ "proportional" , "proportional_estimate_B", "domaingap_22_B_high", "domaingap_22_empirical_B_high" ]
MultiClassAcsCertificates.Plots.acquisition("domaingap_22_empirical_B_high", snames; ternary=false)

snames = [ "proportional" , "proportional_estimate_C", "domaingap_22_C_low", "domaingap_22_empirical_C_low" ]
MultiClassAcsCertificates.Plots.acquisition("domaingap_22_empirical_C_low", snames; ternary=true)

snames = [ "proportional" , "proportional_estimate_C", "domaingap_22_C_high", "domaingap_22_empirical_C_high" ]
MultiClassAcsCertificates.Plots.acquisition("domaingap_22_empirical_C_high", snames; ternary=true)




# # # # # # # # # # # # # # # # # # # # # 
# # ternary Plots for class proportions #
# # # # # # # # # # # # # # # # # # # # # 

# snames = ["domaingap_1Inf_A_high", "domaingap_1Inf_A_low", "proportional"]
# df_agg = MultiClassAcsCertificates.Plots.acquisition("1Inf_A", snames; ternary=true)

# snames = ["domaingap_1Inf_B_high", "domaingap_1Inf_B_low", "proportional_estimate_B"]
# df_agg = MultiClassAcsCertificates.Plots.acquisition("1Inf_B", snames; ternary=true)

# snames = ["domaingap_1Inf_C_high", "domaingap_1Inf_C_low", "proportional_estimate_C"]
# df_agg = MultiClassAcsCertificates.Plots.acquisition("1Inf_C", snames; ternary=true)


# snames = ["domaingap_22_A_high", "domaingap_22_A_low", "proportional"]
# df_agg = MultiClassAcsCertificates.Plots.acquisition("22_A", snames; ternary=true)

# snames = ["domaingap_22_B_high", "domaingap_22_B_low", "proportional_estimate_B"]
# df_agg = MultiClassAcsCertificates.Plots.acquisition("22_B", snames; ternary=true)

# snames = ["domaingap_22_C_high", "domaingap_22_C_low", "proportional_estimate_C"]
# df_agg = MultiClassAcsCertificates.Plots.acquisition("22_C", snames; ternary=true)


# snames = ["domaingap_Inf1_A_high", "domaingap_Inf1_A_low", "proportional"]
# df_agg = MultiClassAcsCertificates.Plots.acquisition("Inf1_A", snames; ternary=true)

# snames = ["domaingap_Inf1_B_high", "domaingap_Inf1_B_low", "proportional_estimate_B"]
# df_agg = MultiClassAcsCertificates.Plots.acquisition("Inf1_B", snames; ternary=true)

# snames = ["domaingap_Inf1_C_high", "domaingap_Inf1_C_low", "proportional_estimate_C"]
# df_agg = MultiClassAcsCertificates.Plots.acquisition("Inf1_C", snames; ternary=true)





