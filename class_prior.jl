using MultiClassAcsCertificates.Data: mle_dirichlet
using MultiClassAcsCertificates.Plots: _ternary_plot_class_prior_distribution, _to_point
using MultiClassAcsCertificates.Experiments: beta_parameters
using Revise, LinearAlgebra
using Distributions: Dirichlet, Beta

# # # # # # #
# Binary 
# # # # # # #
A_expected_value = [0.2, 0.8]
B_expected_value = [0.1, 0.9]
C_expected_value = [0.3, 0.7]

alpha, beta = beta_parameters(A_expected_value[1], 0.05)
d = Beta(alpha, beta)
@info "alpha, beta = $(alpha), $(beta)"
@info "mean d = $(mean(d))"
@info "var d = $(var(d))"

alpha, beta = beta_parameters(A_expected_value[1], 0.2)
d = Beta(alpha, beta)
@info "alpha, beta = $(alpha), $(beta)"
@info "mean d = $(mean(d))"
@info "var d = $(var(d))"

alpha, beta = beta_parameters(B_expected_value[1], 0.05)
d = Beta(alpha, beta)
@info "alpha, beta = $(alpha), $(beta)"
@info "mean d = $(mean(d))"
@info "var d = $(var(d))"

alpha, beta = beta_parameters(B_expected_value[1], 0.2)
d = Beta(alpha, beta)
@info "alpha, beta = $(alpha), $(beta)"
@info "mean d = $(mean(d))"
@info "var d = $(var(d))"

alpha, beta = beta_parameters(C_expected_value[1], 0.05)
d = Beta(alpha, beta)
@info "alpha, beta = $(alpha), $(beta)"
@info "mean d = $(mean(d))"
@info "var d = $(var(d))"

alpha, beta = beta_parameters(C_expected_value[1], 0.2)
d = Beta(alpha, beta)
@info "alpha, beta = $(alpha), $(beta)"
@info "mean d = $(mean(d))"
@info "var d = $(var(d))"



variance_low = 0.01
variance_high = 0.1

A_expected_value = [0.7, 0.2, 0.1]
B_expected_value = [0.8, 0.1, 0.1]
C_expected_value = [0.6, 0.3, 0.1]

savepath="res/plots/acquisition/class_prior/"
pY_T = [_to_point(A_expected_value)]

filename = "A_low"
alpha = mle_dirichlet(A_expected_value, variance_low; n_samples=50000, margin=0.01)
_ternary_plot_class_prior_distribution(Dirichlet(alpha), pY_T, savepath * filename * ".png")
# [ Info: Sampled points mean [0.6680010605432514 0.17629011764752786 0.1557088218092208]
# [ Info: Sampled points variance [0.007362802235364442 0.006233353019768225 0.00920652260832782]
# [ Info: Dirichlet alpha: [14.387542115748195, 3.8703637406512574, 2.7694662073336844]
# [ Info: Dirichlet mean: [0.684229207156278, 0.18406312157888, 0.1317076712648419]
# [ Info: Dirichlet var: [0.009808687055606578, 0.0068180574795296605, 0.005191756886111846]

filename = "A_high"
alpha = mle_dirichlet(A_expected_value, variance_high; n_samples=50000, margin=0.01)
_ternary_plot_class_prior_distribution(Dirichlet(alpha), pY_T, savepath * filename * ".png")
# [ Info: Sampled points mean [0.48938344164953784 0.23296105603329553 0.2776555023171667]
# [ Info: Sampled points variance [0.04263974111174187 0.025757304232941185 0.0377365219279379]
# [ Info: Dirichlet alpha: [7.899646338259085, 2.084806890568132, 2.0547580922750104]
# [ Info: Dirichlet mean: [0.6561597871791363, 0.17316806184088657, 0.17067215097997723]
# [ Info: Dirichlet var: [0.017302742881622025, 0.010980793291342572, 0.01085519395109245]


filename = "B_low"
alpha = mle_dirichlet(B_expected_value, variance_low; n_samples=50000, margin=0.01)
_ternary_plot_class_prior_distribution(Dirichlet(alpha), pY_T, savepath * filename * ".png")
# [ Info: Sampled points mean [0.7597054904382542 0.10557974638400114 0.1347147631777447]
# [ Info: Sampled points variance [0.006711856310816614 0.003773480424389646 0.007044176531694948]
# [ Info: Dirichlet alpha: [17.161071363877603, 2.2666703557889467, 2.62379914012208]
# [ Info: Dirichlet mean: [0.7782254978458724, 0.10278965856405343, 0.11898484359007418]
# [ Info: Dirichlet var: [0.007487159899557319, 0.004000771367835772, 0.004547524663255025]

filename = "B_high"
alpha = mle_dirichlet(B_expected_value, variance_high; n_samples=50000, margin=0.01)
_ternary_plot_class_prior_distribution(Dirichlet(alpha), pY_T, savepath * filename * ".png")
# [ Info: Sampled points mean [0.5487953496368604 0.19284655148457036 0.2583580988785692]
# [ Info: Sampled points variance [0.040118814515977264 0.01911385512222735 0.03355638352831414]
# [ Info: Dirichlet alpha: [10.740962063330857, 1.7419657976874172, 2.128268842615437]
# [ Info: Dirichlet mean: [0.7351185725026649, 0.11922129535455515, 0.14566013214277992]
# [ Info: Dirichlet var: [0.012473051269604823, 0.0067264271972241035, 0.00797141054650615]

filename = "C_low"
alpha = mle_dirichlet(C_expected_value, variance_low; n_samples=50000, margin=0.01)
_ternary_plot_class_prior_distribution(Dirichlet(alpha), pY_T, savepath * filename * ".png")
# [ Info: Sampled points mean [0.5689976448563834 0.2692209462602539 0.1617814088833627]
# [ Info: Sampled points variance [0.00757202889581189 0.007309146952256716 0.010165227333689603]
# [ Info: Dirichlet alpha: [14.115751563994916, 6.891906089894918, 3.138224604457167]
# [ Info: Dirichlet mean: [0.5846028491717356, 0.2854278015669712, 0.1299693492612933]
# [ Info: Dirichlet var: [0.009657340928311024, 0.00811102070566302, 0.004496852262018129]

filename = "C_high"
alpha = mle_dirichlet(C_expected_value, variance_high; n_samples=50000, margin=0.01)
_ternary_plot_class_prior_distribution(Dirichlet(alpha), pY_T, savepath * filename * ".png")
# [ Info: Sampled points mean [0.43444390577166503 0.27237793291941326 0.2931781613089217]
# [ Info: Sampled points variance [0.04173123369900821 0.030179326383204506 0.03991935040477174]
# [ Info: Dirichlet alpha: [7.098578975569465, 3.264872072251279, 2.2658393499033465]
# [ Info: Dirichlet mean: [0.5620726701199847, 0.2585158761445249, 0.1794114537354902]
# [ Info: Dirichlet var: [0.018060146672440026, 0.014064225820425871, 0.010801955179456188]