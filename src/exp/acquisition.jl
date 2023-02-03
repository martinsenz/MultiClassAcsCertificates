function acquisition(config_path="conf/exp/acquisition.yml")

    config = parsefile(config_path)
    results_path = config["writepath"]
    config["rskf"]["n_splits"] = 3 
    config["sample_size_multiplier"] = config["rskf"]["n_splits"]
    experiments = expand(config, "data", "strategy", "estimate_pY_T", "clf", "loss", "delta")

    for exp in experiments
        if !(contains(exp["strategy"], "domaingap") || exp["strategy"] == "proportional_estimate")
            exp["estimate_pY_T"] = nothing
        end
        if exp["strategy"] == "proportional_estimate"
            exp["estimate_pY_T"] = (exp["estimate_pY_T"][1][1:1], exp["estimate_pY_T"][2][1]) # no variance 
        end
        exp["name"] = exp["strategy"]
        if contains(exp["strategy"], "domaingap") || exp["strategy"] == "proportional_estimate"
            exp["name"] = exp["name"] * "_" * exp["estimate_pY_T"][1]
        end
    end
    unique!(experiments)
    @info "There are $(length(experiments)) combinations."
    for (i, exp) in enumerate(experiments)
        exp["info"] = "Trial $(i): $(exp["name"]), classifier=$(exp["clf"]) on dataset=$(exp["data"]))"
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

    # instantiate classifier and loss
    clf_args = Dict{String,Any}()
    clf = SkObject(config["clf"], clf_args)
    L = getproperty(LossFunctions, Symbol(config["loss"]))()

    df = DataFrame(
        i_rskf   = Int[], # iteration of the rskf
        name = String[], # strategy name
        batch    = Int[], # number of the ACS acquisition batch
        N_1    = Int[], # number of class 1 training set instances
        N_2    = Int[], # number of class 2 training set instances
        N_3    = Int[], # number of class 3 training set instances
        pY_trn  = Array{Float64, 1}[], 
        L_tst    = Float64[] # training set loss
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
        i_trn = Data.subsample_indices(y_trn, Int.(floor.((ones(length(classes)) ./ length(classes)) * config["initital_trainingssize"])), classes; seed=seeds[i_rskf])

        for batch in 1:config["n_batches"]
            
            # 1) train and evaluate the classifier
            m_trn = Data.class_counts(y_trn[i_trn], classes)
            ScikitLearn.fit!(clf, X_trn[i_trn, :], y_trn[i_trn])
            y_h_tst = ScikitLearn.predict(clf, X_tst)
            L_tst = sum(Certification.empirical_classwise_risk(L, y_h_tst, y_tst, classes) .* pY_tst)
            # loss check: @info Util.zero_one_sklearn(y_tst, y_h_tst)

            # 2) store an log information ...
            pY_trn = Data.class_proportion(y_trn[i_trn], classes)
            push!(df, [i_rskf, config["name"], batch, m_trn[1], m_trn[2],m_trn[3], pY_trn, L_tst])

            # 3) acquire new data
            if batch < config["n_batches"]
                y_h_trn = ScikitLearn.predict(clf, X_trn[i_trn, :])
                m_d = _m_d(L, y_h_trn, y_trn[i_trn], config)
                m_s = _sanitize_m_d(m_d, config)
                try
                    i_trn = _acquire(i_trn, y_trn, m_s)
                catch some_error
                    if isa(some_error, BoundsError)
                        m_rem = Data._m_y(y_trn[setdiff(1:length(y_trn), i_trn)])
                        if any(m_rem .< m_s) # not enough data remaining; no need to complain
                            #_progress_acquisition!(config, config["n_batches"]-batch) # remaining steps
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

# gibt Strategyempfehlung zurück in Form von Batches
function _m_d(L, y_h, y, config)
    if config["strategy"] == "uniform"
        return fill(config["batchsize"] / 3, 3)
    elseif config["strategy"] == "random"
        rand_i = rand(1:3, config["batchsize"])
        vcat(sum(rand_i .== 1), sum(rand_i .== 2), sum(rand_i .== 3))
    elseif config["strategy"] == "proportional"
        p_d = config["pY_T"]
        m_d = p_d .* (length(y) + config["batchsize"])
        return m_d - Data.class_counts(y, [1,2,3])
    elseif config["strategy"] == "proportional_estimate"
        m_d = config["estimate_pY_T"][2].* (length(y) + config["batchsize"])
        return m_d - Data.class_counts(y, [1,2,3])
    elseif config["strategy"] == "inverse"
        empirical_ℓ_y = min.(1-1e-4, max.(1e-4, empirical_classwise_risk(L, y_h, y, [1,2,3])))
        utility = 1 ./ (1 .- empirical_ℓ_y) # inverse accuracy if L==ZeroOneLoss()
        return utility
    elseif config["strategy"] == "improvement"
        empirical_ℓ_y = min.(1-1e-4, max.(1e-4, empirical_classwise_risk(L, y_h, y, [1,2,3])))
        if config["__cache__"] === nothing # first iteration: inverse strategy
            utility = 1 ./ (1 .- empirical_ℓ_y)
        else
            utility = max.(0, config["__cache__"] .- empirical_ℓ_y) # reduction in loss
        end
        config["__cache__"] = empirical_ℓ_y # store / update losses
        return utility
    elseif config["strategy"] == "redistriction"
        if config["__cache__"] === nothing # first iteration: inverse strategy
            empirical_ℓ_y = min.(1-1e-4, max.(1e-4, empirical_classwise_risk(L, y_h, y, [1,2,3])))
            utility = 1 ./ (1 .- empirical_ℓ_y)
        else
            N = length(config["__cache__"]) # number of samples in previous iteration
            is_redistricted = sign.(y_h[1:N]) .!= config["__cache__"]
            utility = [sum(is_redistricted[y[1:N].==1]), sum(is_redistricted[y[1:N].==2]), sum(is_redistricted[y[1:N].==3])]
        end
        config["__cache__"] = sign.(y_h) # store / update predictions
        return utility
    elseif occursin("domaingap", config["strategy"])
        pac_bounds = !contains(config["strategy"], "empirical")
        plus = contains(config["strategy"], "plus")
        conjugate = _extract_hoelder_conjugate(config["strategy"])
        c = Certification.NormedCertificate(L, y_h, y; hoelder_conjugate=conjugate, pac_bounds=pac_bounds, n_trials=3, delta=config["delta"])
        cov_matrix = Matrix(config["estimate_pY_T"][2][2]*I,3,3)
        class_prior_distribution = Distributions.MvNormal(config["estimate_pY_T"][2][1], cov_matrix)
        Strategy.suggest_acquisition(c, class_prior_distribution, config["batchsize"]; plus=plus)
    else
        throw(ValueError("Unknown strategy \"$strategy\""))
    end


end

function _sanitize_m_d(m_d, config)
    m_s = max.(0.0, m_d)
    m_s[isnan.(m_s)] .= 0
    m_s[m_s .== Inf] .= 1
    if sum(m_s) == 0
        m_s = fill(config["batchsize"]/3, 3)
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

function _acquire(i_trn, y_trn, m_s)
    i_rem = setdiff(1:length(y_trn), i_trn)
    i_acq = i_rem[Data.subsample_indices(y_trn[i_rem], m_s, [1,2,3])]
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