
"""
    tightness(config_path)
    
Performs the tightness experiments from Table 2 and produce trenary plots (included Fig.1, Fig.2)
"""
function tightness(config_path::String="conf/exp/tightness.yml")
    config = parsefile(config_path)
    @info "Read the configuration at $config_path"
    config["rskf"]["n_splits"] = 3 
    config["sample_size_multiplier"] = config["rskf"]["n_splits"]
    trial_configs = expand(config, "data", "clf", "weight", "loss", "delta", "method")
    @info "There are $(length(trial_configs)) combinations."
    for (i, config) in enumerate(trial_configs)
        config["trial_nr"] = i
    end
    df = vcat(pmap(exp -> _tightness_trial(exp), trial_configs)...)
    @info "Writing results to $(config["writepath"])"
    CSV.write(config["writepath"], df)
    df
end

# single trial
function _tightness_trial(config)
    # set up experiments
    clf_name = config["clf"][((findlast(".", config["clf"])...)+1):end]
    @info "Trial $(config["trial_nr"]): $(config["method"]) with classifier=$(clf_name), loss=$(config["loss"]) and δ=$(config["delta"]) on dataset=$(config["data"])"
    rskf = SkObject("sklearn.model_selection.RepeatedStratifiedKFold", config["rskf"])
    Random.seed!(config["rskf"]["random_state"]) 
    clf = SkObject(config["clf"])

    # load dataset 
    d = Data.dataset(config["data"])
    X = Data.X_data(d)
    y = Data.y_data(d)
    classes = Data.classes(d)

    # generate random test points 
    dirichlet = config["pY_tst"]
    pY_T = Data.dirichlet_pY(dirichlet["n_samples"]; α=fill(0.5, length(classes)), margin=dirichlet["margin"], seed=dirichlet["seed"])

    # result dataframe 
    df = DataFrame(
        i_rskf=Int[],
        dataset=String[], # identifier of the dataset
        loss=String[], # loss function
        clf=String[], # classifier 
        delta=Float64[], # specifies the correctness of the certificate
        weight=String[], # class weights (default: uniform)
        method=String[], # certification method
        pY_S=Array{Float64, 1}[], # class proportion of the source domain
        pY_T=Array{Float64, 1}[], # class proportion of the target domain
        emp_loss_val=Float64[], # empirical loss of the source domain
        emp_loss_tst=Float64[], # empirical loss of the target domain 
        ϵ_val=Float64[], # estimation error source data
        ϵ_tst=Float64[], # estimation error target data 
        ℓNorm=Float64[], # empirical classwise loss 
        ℓNormBounded=Float64[], # upper bounded classwise loss 
        ϵ_cert=Float64[] # from the certificate predicted domain induced error
    )

    for (i_rskf, (trn_val, tst)) in enumerate(rskf.split(X, y))
        trn, val = _stratified_split(X, y, trn_val)
        trn.+=1
        trn_val.+=1
        val.+=1
        tst.+=1

        # weights
        pY_S = Data.class_proportion(y[trn_val], classes)
        w_y = _class_weights(pY_S, config["weight"])
        w_trn = MultiClassAcsCertificates.Util.compute_sample_weight(Dict(zip(1:length(classes), w_y)), y[trn])

        # train classifier
        ScikitLearn.fit!(clf, X[trn,:], y[trn]; sample_weight=w_trn)
        y_h_val = ScikitLearn.predict(clf, X[val,:])
        y_h_tst = ScikitLearn.predict(clf, X[tst,:])

        # estimate source loss
        L = getproperty(LossFunctions, Symbol(config["loss"]))()
        L_S_empirical = sum(Certification.empirical_classwise_risk(L, y_h_val, y[val], classes) .* w_y .* pY_S)
        ϵ_val = Certification._ϵ(length(y[val]) * config["sample_size_multiplier"], config["delta"])

        # set up certification method
        hoelder_conjugate = ""
        if occursin("Inf_1", config["method"])
            hoelder_conjugate = "Inf_1"
        elseif occursin("2_2", config["method"])
            hoelder_conjugate = "2_2"
        elseif occursin("1_Inf", config["method"])
            hoelder_conjugate = "1_Inf"
        else 
            @error "Methode name $(config["method"]) not recognized!"
        end
        variant_plus = occursin("Plus", config["method"]) ? true : false # variant_plus = true => |d_(+)|_{∞} * |l|_{1}
        certificate = NormedCertificate(L, y_h_val, y[val]; hoelder_conjugate=hoelder_conjugate, δ=config["delta"], classes=classes, w_y=w_y)
        ℓNormBounded = Inf
        ℓNorm = Inf

        # test the certificate for a variety of points
        for pY_tst in eachrow(pY_T)
            # predict the domain induced error for a given label shift
            ϵ_cert = domaingap_error(certificate, pY_S, pY_tst; variant_plus=variant_plus, pac_bounds=config["pac_bounds"])
            ℓNorm = certificate.ℓNorm
            if config["pac_bounds"]
                ℓNormBounded = certificate.ℓNormBounded
            end
            # subsample test data to match test point pY_tst
            i_pY = Data.subsample_indices(y[tst], Data._m_y_from_proportion(pY_tst, length(y[tst])), classes)
            y_pY = y[tst][i_pY]
            y_h_pY = y_h_tst[i_pY]
            w_y_tst = _class_weights(pY_tst, config["weight"])
            L_T_empirical = sum(Certification.empirical_classwise_risk(L, y_h_pY, y_pY, classes) .* w_y_tst .* pY_tst)
            δ_tst = config["delta"] * 2 # needed for a fair comparison
            ϵ_tst = Certification._ϵ(length(y_pY) * config["sample_size_multiplier"], δ_tst)
            
            # update results 
            df_row = [i_rskf, config["data"], config["loss"], config["clf"], config["delta"], config["weight"], config["method"],
                        pY_S, pY_tst, L_S_empirical, L_T_empirical, ϵ_val, ϵ_tst, ℓNorm, ℓNormBounded, ϵ_cert ]
        
            push!(df, df_row)
        end
    end
    df
end

function _stratified_split(X, y, trn_val)
    rskf = SkObject("sklearn.model_selection.RepeatedStratifiedKFold"; n_splits=2, n_repeats=1)
    trn, val = rskf.split(X[trn_val.+1,:], y[trn_val.+1]).__next__() # first item of generator
    return trn_val[trn.+1], trn_val[val.+1] # trn, val
end

function _class_weights(pY, weight)
    ones = fill(1., length(pY))
    w_y = Dict(
        "uniform" => ones,
        "proportional" => ones ./ pY,
        "sqrt" => sqrt.(ones ./ pY)
    )[weight]
    return w_y ./ maximum(w_y) # w_y ∈ [0, 1]
end