function acquisition(filename::String, strategy_selection::Vector{String}; 
            df_path::String="res/experiments/acquisition.csv", base_output_dir="res/plots/acquisition/", standalone=true, ternary=false)
    
    df = CSV.read(df_path, DataFrame)
    idx = map(strategy -> strategy ∈ strategy_selection ? true : false, df[!, "name"])
    df = df[idx, :]

    df[!, "pY_trn"] = eval.(Meta.parse.(df[!, "pY_trn"]))
    df[!, "pY_T"] = eval.(Meta.parse.(df[!, "pY_T"]))
    df[!, "kl_unif"] = map(i -> Distances.kl_divergence(df[!, "pY_trn"][i], ones(3) ./ 3), 1:nrow(df))
    df[!, "kl_prop"] = map(i -> Distances.kl_divergence(df[!, "pY_trn"][i], df[!, "pY_T"][i] ), 1:nrow(df))
    count_strategies = length(unique(df[!, "name"]))
    @info "There were identified $(count_strategies) strategy configurations" 

    # average test error over all CV repetitions
    gid = ["batch", "clf", "loss", "delta"]
    df = combine(
        groupby(df, vcat(gid, ["name", "pY_T", "data"])),
        "L_tst" => StatsBase.mean => "L_tst",
        "kl_unif" => StatsBase.mean => "kl_unif",
        "kl_prop" => StatsBase.mean => "kl_prop",
        "pY_trn" => x -> mean(x,dims=1)
    )
    
    # cnt = filter(
    #     "nrow" => x -> x .== length(unique(df[!, "name"])),
    #     combine(groupby(df, ["data", "batch", "pY_T"]), nrow)
    # )
    # df = semijoin(df, cnt, on=["data", "batch", "pY_T"])
    df[!, "name"] = map(x -> _mapping_names(x), df[!, "name"])

    if ternary
        _plot_ternary_class_proportions(df, base_output_dir * "/ternary/" * filename * ".png")
    else 
        _plot_critical_diagram(df, base_output_dir * "/tex/" * filename * "_CD.tex", count_strategies; gid=gid, standalone=standalone)
        _plot_kl_diagram(df, base_output_dir * "/tex/" *  filename * "_KL.tex"; gid=gid, standalone=standalone)
    end

    # for f in *.tex; do pdflatex $f; latexmk -c $f; done
end

function _plot_critical_diagram(df, output_path, count_strategies; gid=["batch", "clf", "loss", "delta"], standalone=true)
    sequence = Pair{String, Vector{Pair{String, Vector}}}[]
    for (key, sdf) in pairs(groupby(df, gid))
        n_data = length(unique(sdf[!, :data]))
        if key.batch ∉ 2:9
            continue
        end
        @info "Batch $(key.batch) is based on $(n_data) data sets"
        title = string(key.batch)
        pairs = CriticalDifferenceDiagrams._to_pairs(sdf, :name, :data, :L_tst)
        push!(sequence, title => pairs)
    end
    plot = CriticalDifferenceDiagrams.plot(sequence...)
    plot.style = join([
        "y dir=reverse",
        "ytick={1,2,3,4,5,6,7,8}",
        "yticklabels={2,3,4,5,6,7,8,9}",
        "ylabel={ACS-Batch}",
        "xlabel={avg. Rang}",
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
        "cycle list={{tu01,mark=*},{tu02,mark=square*},{tu03,mark=triangle*},{tu04,mark=diamond*},{tu05,mark=pentagon*},{gray, mark=diamond*}}",
        "width=\\axisdefaultwidth, height=\\axisdefaultheight"
    ], ", ")

    PGFPlots.resetPGFPlotsPreamble()
    PGFPlots.pushPGFPlotsPreamble(join([
        "\\usepackage{amsmath}",
        "\\usepackage{amssymb}",
        "\\definecolor{tu01}{HTML}{84B818}",
        "\\definecolor{tu02}{HTML}{D18B12}",
        "\\definecolor{tu03}{HTML}{1BB5B5}",
        "\\definecolor{tu04}{HTML}{F85A3E}",
        "\\definecolor{tu05}{HTML}{4B6CFC}",
        "\\definecolor{chartreuse(traditional)}{rgb}{0.87, 1.0, 0.0}"
    ], "\n"))
    PGFPlots.save(output_path, plot)
end

function _plot_kl_diagram(df, output_path; gid=["batch", "clf", "loss", "delta"], standalone=true)

    plot = Axis(style = join([
        "title={KL-Divergenz nach \$p_\\mathcal{T}=[0.7, 0.2, 0.1]\$}",
        "legend style={draw=none,fill=none,at={(1.1,.5)},anchor=west,row sep=.25em}",
        "cycle list={{tu01,mark=*},{tu02,mark=square*},{tu03,mark=triangle*},{tu04,mark=diamond*},{tu05,mark=pentagon*}}",
        "xtick={1,3,5,7}"
    ], ", "))
    for (key, sdf) in pairs(groupby(df, vcat(setdiff(gid, ["batch"]), ["name"])))
        if key.name == "proportional"
            continue # KL divergence is usually zero, so that ymode=log does not work in general
        end
        sdf = combine(
            groupby(
                sdf[sdf[!, "batch"].<=8, :],
                vcat(gid, ["name"])
            ),
            "kl_prop" => StatsBase.mean => "kl_prop",
            "kl_prop" => StatsBase.std => "kl_prop_std"
        )
        push!(plot, PGFPlots.Plots.Linear(
            sdf[!, "batch"],
            sdf[!, "kl_prop"],
            legendentry=string(key.name),
            errorBars=PGFPlots.ErrorBars(y=sdf[!, "kl_prop_std"])
        ))
    end
    if standalone
        PGFPlots.resetPGFPlotsPreamble()
        PGFPlots.pushPGFPlotsPreamble(join([
            "\\usepackage{amsmath}",
            "\\usepackage{amssymb}",
            "\\definecolor{tu01}{HTML}{84B818}",
            "\\definecolor{tu02}{HTML}{D18B12}",
            "\\definecolor{tu03}{HTML}{1BB5B5}",
            "\\definecolor{tu04}{HTML}{F85A3E}",
            "\\definecolor{tu05}{HTML}{4B6CFC}",
            "\\definecolor{chartreuse(traditional)}{rgb}{0.87, 1.0, 0.0}"
        ], "\n"))
    end
    PGFPlots.save(output_path, plot)
end

function _plot_ternary_class_proportions(df, output_path; number_batch=10)
    df_agg = combine(groupby(df, ["name", "batch"]), "pY_trn_function" => x -> mean(x,dims=1))
    scale = 100
    figure, tax = Plots.ternary.figure(scale=scale)
    # figure.suptitle("Title", fontsize=20)
    figure.set_size_inches(10, 10)
    tax.boundary(linewidth=3.0)
    tax.gridlines(multiple=scale/10, color="black")
    fontsize = 17
    cb_kwargs = Dict(Dict(:use_gridspec => false, :location => "bottom", :pad => -0.03))

    tax.scatter([_to_point([0.3333,0.3333,0.3333])], marker="D", label="Initial", color="blue", zorder=5)
    tax.scatter([_to_point([0.7,0.2,0.1])], marker="D", label=L"$\mathbf{p}_{T}$", color="red", zorder=5) 
    # style = ["-", "--", ":"]
    #style = ["-", ":", "dotted"]
    #alpha = [.3, 1., 1.]
    for gdf in groupby(df_agg, "name")
        label = gdf[!,"name"][1]
        # label = label[1]
        points = map(p -> MultiClassAcsCertificates.Plots._to_point(gdf[!, "pY_trn_function_function"][p]), 1:number_batch)
        linestyle = "dotted"
        alpha = 1.0
        if contains(label, "proportional")
            linestyle = "-"
            alpha = .3
        end
        tax.plot(points, linestyle=linestyle, marker="x", label=label, alpha=alpha)
    end
    tax.left_axis_label("C3", fontsize=fontsize, position=(-0.10,0.3), rotation=0.0)
    tax.right_axis_label("C2", fontsize=fontsize, position=(0.17,0.96), rotation=0.0)
    tax.bottom_axis_label("C1", fontsize=fontsize, position=(0.79,0.05))
    ticks_labels = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    tax.ticks(ticks=ticks_labels)
    tax.legend(loc="upper left", fontsize=10)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    tax.savefig(output_path, transparent=true, pad_inches=0.0, bbox_inches="tight")
end

function _mapping_names(name)
    if name == "proportional_estimate_B"
        L"$\mathrm{proportional}_{\mathbb{E}_{B}}$"
    elseif name == "proportional_estimate_C"
        L"$\mathrm{proportional}_{\mathbb{E}_{C}}$"
    elseif name == "domaingap_1Inf_A_low"
        L"$\mathrm{domaingap}(\infty, 1)_{\mathbb{E}_{A}}^{\sigma_{low}}$"
    elseif name == "domaingap_1Inf_A_high"
        L"$\mathrm{domaingap}(\infty, 1)_{\mathbb{E}_{A}}^{\sigma_{high}}$"
    elseif name == "domaingap_1Inf_B_low"
        L"$\mathrm{domaingap}(\infty, 1)_{\mathbb{E}_{B}}^{\sigma_{low}}$"
    elseif name == "domaingap_1Inf_B_high"
        L"$\mathrm{domaingap}(\infty, 1)_{\mathbb{E}_{B}}^{\sigma_{high}}$"
    elseif name == "domaingap_1Inf_C_low"
        L"$\mathrm{domaingap}(\infty, 1)_{\mathbb{E}_{C}}^{\sigma_{low}}$"
    elseif name == "domaingap_1Inf_C_high"
        L"$\mathrm{domaingap}(\infty, 1)_{\mathbb{E}_{C}}^{\sigma_{high}}$"

    elseif name == "domaingap_1Inf_empirical_A_low"
        L"$\mathrm{domaingap}_{noPAC}(\infty, 1)_{\mathbb{E}_{A}}^{\sigma_{low}}$"
    elseif name == "domaingap_1Inf_empirical_A_high"
        L"$\mathrm{domaingap}_{noPAC}(\infty, 1)_{\mathbb{E}_{A}}^{\sigma_{high}}$"
    elseif name == "domaingap_1Inf_empirical_B_low"
        L"$\mathrm{domaingap}_{noPAC}(\infty, 1)_{\mathbb{E}_{B}}^{\sigma_{low}}$"
    elseif name == "domaingap_1Inf_empirical_B_high"
        L"$\mathrm{domaingap}_{noPAC}(\infty, 1)_{\mathbb{E}_{B}}^{\sigma_{high}}$"
    elseif name == "domaingap_1Inf_empirical_C_low"
        L"$\mathrm{domaingap}_{noPAC}(\infty, 1)_{\mathbb{E}_{C}}^{\sigma_{low}}$"
    elseif name == "domaingap_1Inf_empirical_C_high"
        L"$\mathrm{domaingap}_{noPAC}(\infty, 1)_{\mathbb{E}_{C}}^{\sigma_{high}}$"

    elseif name == "domaingap_Inf1_A_low"
        L"$\mathrm{domaingap}(1, \infty)_{\mathbb{E}_{A}}^{\sigma_{low}}$"
    elseif name == "domaingap_Inf1_A_high"
        L"$\mathrm{domaingap}(1, \infty)_{\mathbb{E}_{A}}^{\sigma_{high}}$"
    elseif name == "domaingap_Inf1_B_low"
        L"$\mathrm{domaingap}(1, \infty)_{\mathbb{E}_{B}}^{\sigma_{low}}$"
    elseif name == "domaingap_Inf1_B_high"
        L"$\mathrm{domaingap}(1, \infty)_{\mathbb{E}_{B}}^{\sigma_{high}}$"
    elseif name == "domaingap_Inf1_C_low"
        L"$\mathrm{domaingap}(1, \infty)_{\mathbb{E}_{C}}^{\sigma_{low}}$"
    elseif name == "domaingap_Inf1_C_high"
        L"$\mathrm{domaingap}(1, \infty)_{\mathbb{E}_{C}}^{\sigma_{high}}$"

    elseif name == "domaingap_Inf1_empirical_A_low"
        L"$\mathrm{domaingap}_{noPAC}(1, \infty)_{\mathbb{E}_{A}}^{\sigma_{low}}$"
    elseif name == "domaingap_Inf1_empirical_A_high"
        L"$\mathrm{domaingap}_{noPAC}(1, \infty)_{\mathbb{E}_{A}}^{\sigma_{high}}$"
    elseif name == "domaingap_Inf1_empirical_B_low"
        L"$\mathrm{domaingap}_{noPAC}(1, \infty)_{\mathbb{E}_{B}}^{\sigma_{low}}$"
    elseif name == "domaingap_Inf1_empirical_B_high"
        L"$\mathrm{domaingap}_{noPAC}(1, \infty)_{\mathbb{E}_{B}}^{\sigma_{high}}$"
    elseif name == "domaingap_Inf1_empirical_C_low"
        L"$\mathrm{domaingap}_{noPAC}(1, \infty)_{\mathbb{E}_{C}}^{\sigma_{low}}$"
    elseif name == "domaingap_Inf1_empirical_C_high"
        L"$\mathrm{domaingap}_{noPAC}(1, \infty)_{\mathbb{E}_{C}}^{\sigma_{high}}$"
    
    elseif name == "domaingap_22_A_low"
        L"$\mathrm{domaingap}(2, 2)_{\mathbb{E}_{A}}^{\sigma_{low}}$"
    elseif name == "domaingap_22_A_high"
        L"$\mathrm{domaingap}(2, 2)_{\mathbb{E}_{A}}^{\sigma_{high}}$"
    elseif name == "domaingap_22_B_low"
        L"$\mathrm{domaingap}(2, 2)_{\mathbb{E}_{B}}^{\sigma_{low}}$"
    elseif name == "domaingap_22_B_high"
        L"$\mathrm{domaingap}(2, 2)_{\mathbb{E}_{B}}^{\sigma_{high}}$"
    elseif name == "domaingap_22_C_low"
        L"$\mathrm{domaingap}(2, 2)_{\mathbb{E}_{C}}^{\sigma_{low}}$"
    elseif name == "domaingap_22_C_high"
        L"$\mathrm{domaingap}(2, 2)_{\mathbb{E}_{C}}^{\sigma_{high}}$"

    elseif name == "domaingap_22_empirical_A_low"
        L"$\mathrm{domaingap}_{noPAC}(2, 2)_{\mathbb{E}_{A}}^{\sigma_{low}}$"
    elseif name == "domaingap_22_empirical_A_high"
        L"$\mathrm{domaingap}_{noPAC}(2, 2)_{\mathbb{E}_{A}}^{\sigma_{high}}$"
    elseif name == "domaingap_22_empirical_B_low"
        L"$\mathrm{domaingap}_{noPAC}(2, 2)_{\mathbb{E}_{B}}^{\sigma_{low}}$"
    elseif name == "domaingap_22_empirical_B_high"
        L"$\mathrm{domaingap}_{noPAC}(2, 2)_{\mathbb{E}_{B}}^{\sigma_{high}}$"
    elseif name == "domaingap_22_empirical_C_low"
        L"$\mathrm{domaingap}_{noPAC}(2, 2)_{\mathbb{E}_{C}}^{\sigma_{low}}$"
    elseif name == "domaingap_22_empirical_C_high"
        L"$\mathrm{domaingap}_{noPAC}(2, 2)_{\mathbb{E}_{C}}^{\sigma_{high}}$"

    else
        name
    end
end