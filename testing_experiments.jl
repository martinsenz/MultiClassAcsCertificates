using Revise, MultiClassAcsCertificates

# performing tightness experiments (test configurations)
MultiClassAcsCertificates.tightness("conf/exp/test_tightness_binary.yml")
MultiClassAcsCertificates.tightness("conf/exp/test_tightness_multiclass.yml")


# performing acquisition experiments 
MultiClassAcsCertificates.acquisition("conf/exp/test_acquisition_binary.yml")
MultiClassAcsCertificates.acquisition("conf/exp/test_acquisition_multiclass.yml")

# some evaluation plots...
# binary case 
loadpath = "res/experiments/test_acquistion_binary.csv"
filename = "motivation_binary"
snames = [ "inverse", "improvement", "redistriction", "proportional" ]
df = MultiClassAcsCertificates.Plots.acquisition(filename, snames, loadpath)

filename = "proportional_binary"
snames = [ "proportional", "proportional_estimate_B", "binary_certificate_B_low", "binary_certificate_B_high" ]
MultiClassAcsCertificates.Plots.acquisition(filename, snames, loadpath)

# multiclass 
loadpath = "res/experiments/test_acquisition_multiclass.csv"
filename = "motivation_multiclass"
snames = [ "inverse", "improvement", "redistriction", "proportional" ]
MultiClassAcsCertificates.Plots.acquisition(filename, snames, loadpath)

filename = "proportional_multiclass"
snames = [ "proportional", "proportional_estimate_B", "domaingap_plus_1Inf_B_low", "domaingap_plus_1Inf_B_high" ]
MultiClassAcsCertificates.Plots.acquisition(filename, snames, loadpath)

filename = "compare_hoelder"
snames = [ "proportional", "domaingap_plus_1Inf_B_high", "domaingap_plus_22_B_high", "domaingap_plus_Inf1_B_high"]
MultiClassAcsCertificates.Plots.acquisition(filename, snames, loadpath)