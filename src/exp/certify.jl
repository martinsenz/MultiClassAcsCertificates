"""
    certify(config_path)

Performs the certify experiments and generates Table 1 from the results.
"""
function certify(config_path::String="conf/exp/certify.yml")
    config = parsefile(config_path)
    @info "Read the configuration at $config_path"
    config["rskf"]["n_splits"] = 3 
    config["sample_size_multiplier"] = config["rskf"]["n_splits"]
    trial_configs = expand(config, "data", "clf", "weight", "loss", "delta", "method")
    @info "There are $(length(trial_configs)) combinations."
    for (i, config) in enumerate(trial_configs)
        config["trial_nr"] = i
    end
    df = vcat(pmap(exp -> _certify_trial(exp), trial_configs)...)
    @info "Writing results to $(config["writepath"])"
    CSV.write(config["writepath"], df)
    df
end

# single trial
function _certify_trial(config)
    # setup experiment
    clf_name = config["clf"][((findlast(".", config["clf"])...)+1):end]
    @info "Trial $(config["trial_nr"]): $(config["method"]) with classifier=$(clf_name), loss=$(config["loss"]) (weight=$(config["weight"])) and δ=$(config["delta"]) on dataset=$(config["data"])"
    rskf = SkObject("sklearn.model_selection.RepeatedStratifiedKFold", config["rskf"])
    clf = SkObject(config["clf"])
    Random.seed!(config["rskf"]["random_state"]) 

    # load dataset
    d = Data.dataset(config["data"])
    X = Data.X_data(d)
    y = Data.y_data(d)
    classes = Data.classes(d)
    
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
        L_S=Float64[], # loss of the Source domain
        epsilon=Float64[], # controlled domain induced error
        delta_p=Float64[] # Δp^{*} 
    )

    for (i_rskf, (trn_val, tst)) in enumerate(rskf.split(X, y))
        # split data into training, validation and test data
        trn, val = _stratified_split(X, y, trn_val)
        trn.+=1; trn_val.+=1; val.+=1; tst.+=1
        pY_S = Data.class_proportion(y[trn_val], classes) # class proportion source domain

        # weights
        w_y = _class_weights(pY_S, config["weight"])
        w_trn = Util.compute_sample_weight(Dict(zip(1:length(classes), w_y)), y[trn])

        # train classifier
        ScikitLearn.fit!(clf, X[trn,:], y[trn]; sample_weight=w_trn)
        y_h_val = ScikitLearn.predict(clf, X[val,:])

        # instantiate loss and certification method
        L = getproperty(LossFunctions, Symbol(config["loss"]))() 
        L_S_empirical = sum(Certification.empirical_classwise_risk(L, y_h_val, y[val], classes) .* w_y .* pY_S) # empirical estimate loss of the source domain
        ϵ_val = Certification._ϵ(length(y[val]) * config["sample_size_multiplier"], config["delta"]) # onesided estimation error from pac bounds 
        L_S = L_S_empirical + ϵ_val # the upper bounded source loss 

        # certify trained classifier on dataset
        certificate = MultiClassCertificate(L, y_h_val, y[val]; δ=config["delta"], classes=classes, w_y=w_y)
        for eps in config["epsilon"]
            df_row = [i_rskf, config["data"], config["loss"], config["clf"], config["delta"], config["weight"], config["method"], pY_S, L_S, eps, Certification.max_Δp(certificate, eps)]
            push!(df, df_row)
        end
    end
    df
end

