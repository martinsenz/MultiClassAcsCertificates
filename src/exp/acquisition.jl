function acquisition(config_path::String)

    # prepare the experiment configurations
    config = parsefile(config_path)
    results_path = config["writepath"]
    config["rskf"]["n_splits"] = 3 
    config["sample_size_multiplier"] = config["rskf"]["n_splits"]
    experiments = expand(config, "data", "strategy", "estimate_pY_T", "clf", "loss", "delta")
    for exp in experiments
        if !(contains(exp["strategy"], "domaingap") || exp["strategy"] == "proportional_estimate" || exp["strategy"] == "binary_certificate")
            exp["estimate_pY_T"] = nothing
        end
        if exp["strategy"] == "proportional_estimate"
            estimate_pY_T = begin
                if length(exp["pY_T"]) == 2
                    β, α = exp["estimate_pY_T"][2]
                    pY = mean(Distributions.mean(Beta(α, β)))
                    vcat(pY, 1.0 - pY)
                else
                    mean(Distributions.Dirichlet(exp["estimate_pY_T"][2]))
                end
            end
            exp["estimate_pY_T"] = (exp["estimate_pY_T"][1][1:1], estimate_pY_T) # no variance for proportional_estimate 
        end
        exp["name"] = exp["strategy"]
        if contains(exp["strategy"], "domaingap") || exp["strategy"] == "proportional_estimate" || exp["strategy"] == "binary_certificate"
            exp["name"] = exp["name"] * "_" * exp["estimate_pY_T"][1]
        end
    end
    unique!(experiments)
    @info "There are $(length(experiments)) combinations."
    for (i, exp) in enumerate(experiments)
        exp["info"] = "Trial $(i): $(exp["name"]), classifier=$(exp["clf"]) on dataset=$(exp["data"])"
    end
    df = vcat(pmap(exp -> _acquisition(exp), experiments)...)
    @info "Writing results" results_path
    CSV.write(results_path, df)
    df
end

function _acquisition(config)
    @info "$(config["info"])"
    rskf = SkObject("sklearn.model_selection.RepeatedStratifiedKFold", config["rskf"])
    Random.seed!(config["rskf"]["random_state"])

    # load data
    d = Data.dataset(config["data"])
    X = d.X_data
    y = d.y_data
    classes = sort(unique(y))
    config["classes"] = classes
    config["n"] = length(classes)

    # instantiate classifier and loss
    clf_args = Dict{String,Any}()
    clf = SkObject(config["clf"], clf_args)
    L = getproperty(LossFunctions, Symbol(config["loss"]))()

    df = DataFrame(
        i_rskf = Int[], # iteration of the rskf
        name = String[], # strategy name
        batch = Int[], # number of the ACS acquisition batch
        N_1 = Int[], # number of class 1 training set instances
        N_2 = Int[], # number of class 2 training set instances
        N_3 = Int[], # number of class 3 training set instances
        pY_trn = Array{Float64, 1}[], 
        L_tst = Float64[] # training set loss
    )
    seeds = rand(UInt32, rskf.get_n_splits())

    for (i_rskf, (trn, tst)) in enumerate(rskf.split(X, y))
        Random.seed!(seeds[i_rskf])
        config["__cache__"] = nothing 
        trn.+=1
        tst.+=1

        # subsample test set according to pY_T
        i_tst = Data.subsample_indices(y[tst], Data._m_y_from_proportion(config["pY_T"], length(y[tst])), classes; seed=seeds[i_rskf])
        y_tst = y[tst][i_tst]
        X_tst = X[tst,:][i_tst, :]
        pY_tst = Data.class_proportion(y_tst, classes)

        # set up the training data pool
        X_trn = X[trn,:]
        y_trn = y[trn]
       
        # first batch is drawn equally distributed
        i_trn = Data.subsample_indices(y_trn, Int.(floor.((ones(length(classes)) ./ length(classes)) * config["initial_trainingsize"])), classes; seed=seeds[i_rskf])

        for batch in 1:config["n_batches"]
            
            # 1) train and evaluate the classifier
            m_trn = Data.class_counts(y_trn[i_trn], classes)
            if length(m_trn) == 2
                m_trn = vcat(m_trn, 0)
            end
            ScikitLearn.fit!(clf, X_trn[i_trn, :], y_trn[i_trn])
            y_h_tst = ScikitLearn.predict(clf, X_tst)
            L_tst = sum(Certification.empirical_classwise_risk(L, y_h_tst, y_tst, classes) .* pY_tst)

            # 2) store an log information ...
            pY_trn = Data.class_proportion(y_trn[i_trn], classes)
            push!(df, [i_rskf, config["name"], batch, m_trn[1], m_trn[2],m_trn[3], pY_trn, L_tst])

            # 3) acquire new data
            if batch < config["n_batches"]
                y_h_trn = ScikitLearn.predict(clf, X_trn[i_trn, :])
                m_d = _m_d(L, y_h_trn, y_trn[i_trn], config)
                m_s = _sanitize_m_d(m_d, config)
                try
                    i_trn = _acquire(i_trn, y_trn, m_s, classes)
                catch some_error
                    if isa(some_error, BoundsError)
                        m_rem = Data._m_y(y_trn[setdiff(1:length(y_trn), i_trn)])
                        if any(m_rem .< m_s) # not enough data remaining; no need to complain
                            break # stop the ACS loop
                        else
                            @error "BoundsError" batch m_trn m_s m_rem m_tst=Data._m_y(y_tst)
                            rethrow()
                        end
                    else
                        rethrow()
                    end
                end
            end
        end
    end
    for column in [ "data", "clf", "pY_T", "estimate_pY_T", "loss", "delta" ]
        df[!, column] .= string(config[column])
    end
    return df
end

# acquisition suggestions from ACS strategies
function _m_d(L, y_h, y, config)
    if config["strategy"] == "uniform"
        return fill(config["batchsize"] / config["n"], config["n"])
    elseif config["strategy"] == "random"
        rand_i = rand(1:config["n"], config["batchsize"])
        Data.class_counts(rand_i, config["classes"])
    elseif config["strategy"] == "proportional"
        p_d = config["pY_T"]
        m_d = p_d .* (length(y) + config["batchsize"])
        return m_d - Data.class_counts(y, config["classes"])
    elseif config["strategy"] == "proportional_estimate"
        m_d = config["estimate_pY_T"][2] .* (length(y) + config["batchsize"])
        return m_d .- Data.class_counts(y, config["classes"])
    elseif config["strategy"] == "inverse"
        empirical_ℓ_y = min.(1-1e-4, max.(1e-4, empirical_classwise_risk(L, y_h, y, config["classes"])))
        utility = 1 ./ (1 .- empirical_ℓ_y) # inverse accuracy if L==ZeroOneLoss()
        return utility
    elseif config["strategy"] == "improvement"
        empirical_ℓ_y = min.(1-1e-4, max.(1e-4, empirical_classwise_risk(L, y_h, y, config["classes"])))
        if config["__cache__"] === nothing # first iteration: inverse strategy
            utility = 1 ./ (1 .- empirical_ℓ_y)
        else
            utility = max.(0, config["__cache__"] .- empirical_ℓ_y) # reduction in loss
        end
        config["__cache__"] = empirical_ℓ_y # store / update losses
        return utility
    elseif config["strategy"] == "redistriction"
        if config["__cache__"] === nothing # first iteration: inverse strategy
            empirical_ℓ_y = min.(1-1e-4, max.(1e-4, empirical_classwise_risk(L, y_h, y, config["classes"])))
            utility = 1 ./ (1 .- empirical_ℓ_y)
        else
            N = length(config["__cache__"]) # number of samples in previous iteration
            is_redistricted = sign.(y_h[1:N]) .!= config["__cache__"]
            utility = []
            for class in config["classes"]
                push!(utility, sum(is_redistricted[y[1:N].==class]))
            end
        end 
        config["__cache__"] = sign.(y_h) # store / update predictions
        return utility
    elseif occursin("domaingap", config["strategy"])
        pac_bounds = !contains(config["strategy"], "empirical")
        plus = contains(config["strategy"], "plus")
        conjugate = _extract_hoelder_conjugate(config["strategy"])
        c = Certification.NormedCertificate(L, y_h, y; hoelder_conjugate=conjugate, pac_bounds=pac_bounds, n_trials=config["n_trials"], delta=config["delta"])
        class_prior_distribution = Distributions.Dirichlet(config["estimate_pY_T"][2])
        Strategy.suggest_acquisition(c, class_prior_distribution, config["batchsize"]; plus=plus)
    elseif config["strategy"] == "binary_certificate"
        c = Certification.BinaryCertificate(L, Data._binary_relabeling(y_h), Data._binary_relabeling(y); 
            δ=config["delta"],
            warn=config["warn"],
            n_trials=config["n_trials"],
            allow_onesided=config["allow_onesided"], # acquisition certificates must be two-sided
            n_trials_extra=config["n_trials_extra"] # allow more trials if 3 random initializations fail
        )
        β, α = config["estimate_pY_T"][2]
        return suggest_acquisition(c.Δℓ, config["batchsize"], α, β)
    else
        throw(ValueError("Unknown strategy \"$strategy\""))
    end
end

function _sanitize_m_d(m_d, config)
    n_classes = length(m_d)
    m_s = max.(0.0, m_d)
    m_s[isnan.(m_s)] .= 0
    m_s[m_s .== Inf] .= 1
    if sum(m_s) == 0
        m_s = fill(config["batchsize"]/n_classes, n_classes)
    end
    m_s = round.(Int, m_s .* (config["batchsize"] / sum(m_s)))

    if sum(m_s) == config["batchsize"] + 3
        m_s .-= 1
    elseif sum(m_s) == config["batchsize"] + 2
        m_s[findmax(m_s)[2]] -= 1
        m_s[findmax(m_s)[2]] -= 1
    elseif sum(m_s) == config["batchsize"] + 1
        m_s[findmax(m_s)[2]] -= 1
    elseif sum(m_s) == config["batchsize"]-1
        m_s[findmin(m_s)[2]] += 1
    elseif sum(m_s) == config["batchsize"]-2
        m_s[findmin(m_s)[2]] += 1
        m_s[findmin(m_s)[2]] += 1
    elseif sum(m_s) == config["batchsize"]-2
        m_s .+= 1
    end
    if sum(m_s) != config["batchsize"]
        @warn "_sanitize_m_d" m_d m_s sum(m_s)
    end
    return m_s

end

function _acquire(i_trn, y_trn, m_s, classes)
    i_rem = setdiff(1:length(y_trn), i_trn)
    i_acq = i_rem[Data.subsample_indices(y_trn[i_rem], m_s, classes)]
    return vcat(i_trn, i_acq)
end

function _extract_hoelder_conjugate(name::String)
    if contains(name, "1Inf")
        "1_Inf"
    elseif contains(name, "22")
        "2_2"
    elseif contains(name, "Inf1")
        "Inf_1"
    else
        @error("Hoelder conjugate not recognized!. Check the strategy name!")
    end
    
end