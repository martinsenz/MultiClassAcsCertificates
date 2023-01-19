


_strategy_names = Dict([
    ("random", "random"),
    ("uniform", "uniform"),
    ("inverse", "inverse"),
    ("improvement", "improvement"),
    ("redistriction", "redistriction"),
    ("proportional", "proportional"),
    ("proportional_estimate", "prop exp"),
    ("domaingap_1Inf", L"$\lVert \mathbf{d} \rVert_{1} \cdot \lVert \boldsymbol{\ell}_{h} \rVert_{\infty}^{*}$"),
    ("domaingap_22", L"$\lVert \mathbf{d} \rVert_{2} \cdot \lVert \boldsymbol{\ell}_{h} \rVert_{2}^{*}$"),
    ("domaingap_Inf1", L"$\lVert \mathbf{d} \rVert_{\infty} \cdot \lVert \boldsymbol{\ell}_{h} \rVert_{1}^{*}$"),
    ("domaingap_1Inf_empirical", L"$\lVert \mathbf{d} \rVert_{1} \cdot \hat{\lVert \boldsymbol{\ell}_{h} \rVert}_{\infty}$"),
    ("domaingap_22_empirical", L"$\lVert \mathbf{d} \rVert_{2} \cdot \hat{\lVert \boldsymbol{\ell}_{h} \rVert}_{2}$"),
    ("domaingap_Inf1_empirical", L"$\lVert \mathbf{d} \rVert_{\infty} \cdot \hat{\lVert \boldsymbol{\ell}_{h} \rVert}_{1}$")
])

_pY_estimate = Dict([
    ("A", "Any[[0.7, 0.2, 0.1], 0.1]"),
    ("B", "Any[[0.9, 0.05, 0.05], 0.1]"),
    ("C", "Any[[0.8, 0.1, 0.1], 0.1]"),
    ("D", "Any[[0.6, 0.3, 0.1], 0.1]")
])

function _acquisition(pY_T_estimate::String; df_path::String="res/experiments/acquisition.csv", output_path="res/plots/", strategy_selection=collect(keys(_strategy_names)))

    df = CSV.read(df_path, DataFrame)
    df[!, "pY_trn"] = eval.(Meta.parse.(df[!, "pY_trn"]))
    df[!, "pY_T"] = eval.(Meta.parse.(df[!, "pY_T"]))
    df[!, "kl_unif"] = map(i -> Distances.kl_divergence(df[!, "pY_trn"][i], ones(3) ./ 3 ), 1:nrow(df))
    df[!, "kl_prop"] = map(i -> Distances.kl_divergence(df[!, "pY_trn"][i], df[!, "pY_T"][i] ), 1:nrow(df))

    strategy_names = intersect(map(String, unique(df[!, "strategy"])), strategy_selection)
    count_strategies = length(strategy_names)
 
    filter!(row -> row["strategy"] in strategy_names, df)
    df[!, "strategy"] = map(s -> _strategy_names[s], df[!, "strategy"])
    @info "Found $(count_strategies) strategies : \n $(strategy_names)"
    
    idx_pY_T_est = (df[!, "estimate_pY_T"] .== "nothing") .| (df[!, "estimate_pY_T"] .== _pY_estimate[pY_T_estimate])
    df = df[idx_pY_T_est, :]

    # average test error over all CV repetitions
    gid = ["batch", "clf", "loss", "delta", "epsilon"]
    df = combine(
        groupby(df, vcat(gid, ["strategy", "pY_T", "data"])),
        "L_tst" => StatsBase.mean => "L_tst",
        "kl_unif" => StatsBase.mean => "kl_unif",
        "kl_prop" => StatsBase.mean => "kl_prop"
    )

    cnt = filter(
        "nrow" => x -> x .== length(unique(df[!, "strategy"])),
        combine(groupby(df, ["data", "batch", "pY_T"]), nrow)
    )
    df = semijoin(df, cnt, on=["data", "batch", "pY_T"])

    # one CD diagram and one KL divergence diagram per pY_T value
    groupplot = GroupPlot(2, 2, groupStyle = "horizontal sep = 4.0cm")

    for (key_pY, sdf_pY) in pairs(groupby(df, :pY_T))

        # sequence of CD diagrams
        sequence = Pair{String, Vector{Pair{String, Vector}}}[]
        for (key, sdf) in pairs(groupby(sdf_pY, gid))
            n_data = length(unique(sdf[!, :data]))
            if key.batch ∉ 2:9
                continue
            end
            @info "Batch $(key.batch) on pY_T=$(round.(key_pY.pY_T; digits=1)) is based on $(n_data) data sets"
            title = string(key.batch)
            pairs = CriticalDifferenceDiagrams._to_pairs(sdf, :strategy, :data, :L_tst)
            push!(sequence, title => pairs)
        end
  
        plot = CriticalDifferenceDiagrams.plot(sequence...)
        plot.style = join([
            "y dir=reverse",
            "ytick={1,2,3,4,5,6,7,8}",
            "yticklabels={2,3,4,5,6,7,8,9}",
            "ylabel={ACS-Batch}",
            "xlabel={avg. Rang \\, (\$p_\\mathcal{T}=$(round.(key_pY.pY_T; digits=1))\$)}",
            "ylabel style={font=\\small}",
            "xlabel style={font=\\small}",
            "yticklabel style={font=\\small}",
            "xticklabel style={font=\\small}",
            "grid=both",
            "axis line style={draw=none}",
            "tick style={draw=none}",
            "xticklabel pos=upper",
            "xmin=.5",
            "xmax=$(count_strategies + 0.5)",
            "ymin=.75",
            "ymax=$(length(sequence)).75",
            "clip=false",
            "legend style={draw=none,fill=none,at={(1.1,.5)},anchor=west,row sep=.25em}",
            "x dir=reverse",
            "cycle list={{tu01,mark=*},{tu02,mark=square*},{tu03,mark=triangle*},{tu04,mark=diamond*},{tu05,mark=pentagon*}}",
            "width=\\axisdefaultwidth, height=\\axisdefaultheight"
        ], ", ")
        push!(groupplot, plot)

        # KL divergence to proportional sampling
        kl_plot = Axis(style = join([
            "title={KL-Divergenz nach \$p_\\mathcal{T}=$(round.(key_pY.pY_T; digits=1))\$}",
            "legend style={draw=none,fill=none,at={(1.1,.5)},anchor=west,row sep=.25em}",
            "cycle list={{tu01,mark=*},{tu02,mark=square*},{tu03,mark=triangle*},{tu04,mark=diamond*},{tu05,mark=pentagon*}}",
            "xtick={1,3,5,7}"
        ], ", "))
        for (key, sdf) in pairs(groupby(sdf_pY, vcat(setdiff(gid, ["batch"]), ["strategy"])))
            if key.strategy == "proportional"
                continue # KL divergence is usually zero, so that ymode=log does not work in general
            end
            sdf = combine(
                groupby(
                    sdf[sdf[!, "batch"].<=8, :],
                    vcat(gid, ["strategy"])
                ),
                "kl_prop" => StatsBase.mean => "kl_prop",
                "kl_prop" => StatsBase.std => "kl_prop_std"
            )
            push!(kl_plot, PGFPlots.Plots.Linear(
                sdf[!, "batch"],
                sdf[!, "kl_prop"],
                legendentry=string(key.strategy),
                errorBars=PGFPlots.ErrorBars(y=sdf[!, "kl_prop_std"])
            ))
        end
        push!(groupplot, kl_plot)
    end

    PGFPlots.resetPGFPlotsPreamble()
    PGFPlots.pushPGFPlotsPreamble(join([
        "\\usepackage{amsmath}",
        "\\definecolor{tu01}{HTML}{84B818}",
        "\\definecolor{tu02}{HTML}{D18B12}",
        "\\definecolor{tu03}{HTML}{1BB5B5}",
        "\\definecolor{tu04}{HTML}{F85A3E}",
        "\\definecolor{tu05}{HTML}{4B6CFC}",
        "\\definecolor{chartreuse(traditional)}{rgb}{0.87, 1.0, 0.0}"
    ], "\n"))
    @info "Written to $(output_path * pY_T_estimate * ".tex")"
    PGFPlots.save(output_path * pY_T_estimate * ".tex", groupplot)
    return nothing

end

function acquisition(;df_path::String="res/experiments/acquisition.csv", output_path="res/plots/acquisition/", strategy_selection=collect(keys(_strategy_names)))
    for pY_T_estimate in collect(keys(_pY_estimate))
        _acquisition(pY_T_estimate; df_path=df_path, output_path=output_path, strategy_selection=strategy_selection)
    end
end