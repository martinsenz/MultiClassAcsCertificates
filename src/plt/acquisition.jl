
function acquisition(filename::String, strategy_selection::Vector{String}, loadpath;
                base_output_dir="res/plots/acquisition/", standalone=true, color=["tu01", "tu02", "tu03", "tu04", "tu05"])
    
    df = CSV.read(loadpath, DataFrame)
    idx = map(strategy -> strategy ∈ strategy_selection ? true : false, df[!, "name"])
    df = df[idx, :]
    df[!, "pY_trn"] = eval.(Meta.parse.(df[!, "pY_trn"]))
    df[!, "pY_T"] = eval.(Meta.parse.(df[!, "pY_T"]))
    n_classes = length(df[!, "pY_T"][1])
    df[!, "kl_unif"] = map(i -> Distances.kl_divergence(df[!, "pY_trn"][i], ones(n_classes) ./ n_classes), 1:nrow(df))
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
        "pY_trn" => x -> mean(x,dims=1))
    if n_classes == 2
        _plot_kl_diagram(df, base_output_dir * "/tex/" *  filename * "_KL.tex"; gid=gid, standalone=standalone)
        df[!, "name"] = map(x -> _mapping_names_binary(x), df[!, "name"])
    else
        _ternary_batch_proportions(df, base_output_dir * "/ternary/" * filename * "_batch_prop.png"; color=color, strategy_selection=strategy_selection)
        df[!, "name"] = map(x -> _mapping_names_multiclass(x), df[!, "name"])
    end
    _plot_critical_diagram(df, base_output_dir * "/tex/" * filename * "_CD.tex", count_strategies; gid=gid, standalone=standalone)
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
        "ytick={1,2,3,4,5,6,7,8,9}",
        "yticklabels={2,3,4,5,6,7,8,9,10}",
        "ylabel={batch}",
        "xlabel={avg. rank}",
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
        "legend style={draw=none,fill=none,at={(0.5,-0.02)},anchor=north,column sep=.25em, legend columns=4}",
        "x dir=reverse",
        "cycle list={{tu01,mark=*},{tu02,mark=square*},{tu03,mark=triangle*},{tu04,mark=diamond*},{tu05,mark=pentagon*},{gray, mark=diamond*}}",
        "width=\\axisdefaultwidth, height=\\axisdefaultheight"
    ], ", ")
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
    @info "Print CD in $(output_path)"
end

function _plot_kl_diagram(df, output_path; gid=["batch", "clf", "loss", "delta"], standalone=true)

    plot = Axis(style = join([
        "title={}",
        "legend style={draw=none,fill=none,at={(0.5,-0.02)},anchor=north,column sep=.25em, legend columns=4}",
        "cycle list={{tu01,mark=*},{tu02,mark=square*},{tu03,mark=triangle*},{tu04,mark=diamond*},{tu05,mark=pentagon*}}",
        "xtick={1,2,3,4,5,6,7,8,9,10}",
        "axis x line=top",
        "xticklabel pos=top",
        "axis y line=right",
        "yticklabel pos=right",
        "ylabel=\$d_{\text{KL}}\$",
        "xlabel=batch",
        "ylabel style={shift={(0,1ex)}}",
        "xmajorgrids",
        "tick style={draw=none}",
        "clip=false",
        "axis line style={draw=none}",
        "xlabel style={font=\\small, shift={(0,0.7ex)}}",
        "xticklabel style={font=\\small}"
    ], ", "))
    for (key, sdf) in pairs(groupby(df, vcat(setdiff(gid, ["batch"]), ["name"])))
        #if key.name == "proportional"
        #    continue # KL divergence is usually zero, so that ymode=log does not work in general
        #end
        sdf = combine(
            groupby(
                sdf[sdf[!, "batch"].<=10, :],
                vcat(gid, ["name"])
            ),
            "kl_prop" => StatsBase.mean => "kl_prop",
            "kl_prop" => StatsBase.std => "kl_prop_std"
        )
        push!(plot, PGFPlots.Plots.Linear(
            sdf[!, "batch"],
            sdf[!, "kl_prop"],
            legendentry=_mapping_names_binary(string(key.name)),
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

function _ternary_batch_proportions(df, output_path; 
                                        number_batch=10, color=["tu01", "tu02", "tu03", "tu04", "tu05"], strategy_selection=strategy_selection)
    df_agg = combine(groupby(df, ["name", "batch"]), "pY_trn_function" => x -> mean(x,dims=1))
    figure, tax = Plots.ternary.figure(scale=scale)
    figure.set_size_inches(10, 10)
    tax.boundary(linewidth=3.0)
    tax.gridlines(multiple=scale/10, color="black")
    fontsize = 17
    cb_kwargs = Dict(Dict(:use_gridspec => false, :location => "bottom", :pad => -0.03))

    tax.scatter([_to_point([0.3333,0.3333,0.3333])], marker="D", label=L"$\mathbf{p}_\mathcal{S}$", color="blue", zorder=5)
    tax.scatter([_to_point([0.7,0.2,0.1])], marker="D", label=L"$\mathbf{p}_\mathcal{T}$", color="red", zorder=5) 
    i = 1
    for strategy in strategy_selection
        gdf = df_agg[df_agg[!, "name"] .== strategy, :]
        label = _mapping_names_multiclass(gdf[!,"name"][1])
        points = map(p -> MultiClassAcsCertificates.Plots._to_point(gdf[!, "pY_trn_function_function"][p]), 1:number_batch)
        linestyle = "dotted"
        marker = "x"
        alpha = 1.0
        if gdf[!,"name"][1] == "proportional"
            linestyle = "-"
            marker = "."
            alpha = 1.0
        end
        if contains(gdf[!,"name"][1], "estimate")
            linestyle = "-."
            marker = "."
            alpha = 1.0
        end
        tax.plot(points, linestyle=linestyle, marker=marker, color=tu_colors[color[i]], label=label, alpha=alpha)
        i += 1
    end
    tax.left_axis_label("C3", fontsize=fontsize, position=(-0.10,0.3), rotation=0.0)
    tax.right_axis_label("C2", fontsize=fontsize, position=(0.17,0.96), rotation=0.0)
    tax.bottom_axis_label("C1", fontsize=fontsize, position=(0.79,0.05))
    ticks_labels = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    tax.ticks(ticks=ticks_labels)
    tax.legend(loc="upper left", fontsize=17, framealpha=0)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    tax.savefig(output_path, transparent=true, pad_inches=0.0, bbox_inches="tight")
    @info "Print ternary class proportions in $(output_path)"
end

function _mapping_names_binary(name)
    if name == "proportional_estimate_B"
        L"$\mathrm{proportional}_{{o}}$"
    elseif name == "proportional_estimate_C"
        L"$\mathrm{proportional}_{{u}}$"
    elseif name == "proportional"
        L"$\mathrm{proportional}_{{p}_{\mathcal{T}}}$"

    elseif name ∈ ["domaingap_1Inf_A_low", "domaingap_plus_1Inf_A_low"]
        L"$\mathrm{certificate}(1,\infty)_{{p}_{\mathcal{T}}}^{low}$"
    elseif name ∈ ["domaingap_1Inf_A_high", "domaingap_plus_1Inf_A_high"]
        L"$\mathrm{certificate}(1,\infty)_{{p}_{\mathcal{T}}}^{high}$"
    elseif name ∈ ["domaingap_1Inf_B_low", "domaingap_plus_1Inf_B_low"]
        L"$\mathrm{certificate}(1,\infty)_{{o}}^{low}$"
    elseif name ∈ ["domaingap_1Inf_B_high", "domaingap_plus_1Inf_B_high"]
        L"$\mathrm{certificate}(1,\infty)_{{o}}^{high}$"
    elseif name ∈ ["domaingap_1Inf_C_low", "domaingap_plus_1Inf_C_low"]
        L"$\mathrm{certificate}(1,\infty)_{{u}}^{low}$"
    elseif name ∈ ["domaingap_1Inf_C_high", "domaingap_plus_1Inf_C_high"]
        L"$\mathrm{certificate}(1,\infty)_{{u}}^{high}$"

    elseif name ∈ ["domaingap_Inf1_A_low", "domaingap_plus_Inf1_A_low"]
        L"$\mathrm{certificate}(\infty,1)_{{p}_{\mathcal{T}}}^{low}$"
    elseif name ∈ ["domaingap_Inf1_A_high", "domaingap_plus_Inf1_A_high"]
        L"$\mathrm{certificate}(\infty,1)_{{p}_{\mathcal{T}}}^{high}$"
    elseif name ∈ ["domaingap_Inf1_B_low", "domaingap_plus_Inf1_B_low"]
        L"$\mathrm{certificate}(\infty,1)_{{o}}^{low}$"
    elseif name ∈ ["domaingap_Inf1_B_high", "domaingap_plus_Inf1_B_high"]
        L"$\mathrm{certificate}(\infty,1)_{{o}}^{high}$"
    elseif name ∈ ["domaingap_Inf1_C_low", "domaingap_plus_Inf1_C_low"]
        L"$\mathrm{certificate}(\infty,1)_{{u}}^{low}$"
    elseif name ∈ ["domaingap_Inf1_C_high", "domaingap_plus_Inf1_C_high"]
        L"$\mathrm{certificate}(\infty,1)_{{u}}^{high}$"

    elseif name ∈ ["domaingap_22_A_low", "domaingap_plus_22_A_low"]
        L"$\mathrm{certificate}(2,2)_{{p}_{\mathcal{T}}}^{low}$"
    elseif name ∈ ["domaingap_22_A_high", "domaingap_plus_22_A_high"]
        L"$\mathrm{certificate}(2,2)_{{p}_{\mathcal{T}}}^{high}$"
    elseif name ∈ ["domaingap_22_B_low", "domaingap_plus_22_B_low"]
        L"$\mathrm{certificate}(2,2)_{{o}}^{low}$"
    elseif name ∈ ["domaingap_22_B_high", "domaingap_plus_22_B_high"]
        L"$\mathrm{certificate}(2,2)_{{o}}^{high}$"
    elseif name ∈ ["domaingap_22_C_low", "domaingap_plus_22_C_low"]
        L"$\mathrm{certificate}(2,2)_{{u}}^{low}$"
    elseif name ∈ ["domaingap_22_C_high", "domaingap_plus_22_C_high"]
        L"$\mathrm{certificate}(2,2)_{{u}}^{high}$"

    elseif name == "binary_certificate_A_low"
        L"$\mathrm{certificate}_{{p}_{\mathcal{T}}}^{low}$"
    elseif name == "binary_certificate_A_high"
        L"$\mathrm{certificate}_{{p}_{\mathcal{T}}}^{high}$"
    elseif name == "binary_certificate_B_low"
        L"$\mathrm{certificate}_{{o}}^{low}$"
    elseif name == "binary_certificate_B_high"
        L"$\mathrm{certificate}_{{o}}^{high}$"
    elseif name == "binary_certificate_C_low"
        L"$\mathrm{certificate}_{{u}}^{low}$"
    elseif name == "binary_certificate_C_high"
        L"$\mathrm{certificate}_{{u}}^{high}$"
    else
        name
    end
end

function _mapping_names_multiclass(name)
    if name == "proportional_estimate_B"
        L"$\mathrm{proportional}_{\mathbf{o}}$"
    elseif name == "proportional_estimate_C"
        L"$\mathrm{proportional}_{\mathbf{u}}$"
    elseif name == "proportional"
        L"$\mathrm{proportional}_{\mathbf{p}_{\mathcal{T}}}$"

    elseif name ∈ ["domaingap_1Inf_A_low", "domaingap_plus_1Inf_A_low"]
        L"$\mathrm{certificate}(1,\infty)_{\mathbf{p}_{\mathcal{T}}}^{low}$"
    elseif name ∈ ["domaingap_1Inf_A_high", "domaingap_plus_1Inf_A_high"]
        L"$\mathrm{certificate}(1,\infty)_{\mathbf{p}_{\mathcal{T}}}^{high}$"
    elseif name ∈ ["domaingap_1Inf_B_low", "domaingap_plus_1Inf_B_low"]
        L"$\mathrm{certificate}(1,\infty)_{\mathbf{o}}^{low}$"
    elseif name ∈ ["domaingap_1Inf_B_high", "domaingap_plus_1Inf_B_high"]
        L"$\mathrm{certificate}(1,\infty)_{\mathbf{o}}^{high}$"
    elseif name ∈ ["domaingap_1Inf_C_low", "domaingap_plus_1Inf_C_low"]
        L"$\mathrm{certificate}(1,\infty)_{\mathbf{u}}^{low}$"
    elseif name ∈ ["domaingap_1Inf_C_high", "domaingap_plus_1Inf_C_high"]
        L"$\mathrm{certificate}(1,\infty)_{\mathbf{u}}^{high}$"

    elseif name ∈ ["domaingap_Inf1_A_low", "domaingap_plus_Inf1_A_low"]
        L"$\mathrm{certificate}(\infty,1)_{\mathbf{p}_{\mathcal{T}}}^{low}$"
    elseif name ∈ ["domaingap_Inf1_A_high", "domaingap_plus_Inf1_A_high"]
        L"$\mathrm{certificate}(\infty,1)_{\mathbf{p}_{\mathcal{T}}}^{high}$"
    elseif name ∈ ["domaingap_Inf1_B_low", "domaingap_plus_Inf1_B_low"]
        L"$\mathrm{certificate}(\infty,1)_{\mathbf{o}}^{low}$"
    elseif name ∈ ["domaingap_Inf1_B_high", "domaingap_plus_Inf1_B_high"]
        L"$\mathrm{certificate}(\infty,1)_{\mathbf{o}}^{high}$"
    elseif name ∈ ["domaingap_Inf1_C_low", "domaingap_plus_Inf1_C_low"]
        L"$\mathrm{certificate}(\infty,1)_{\mathbf{u}}^{low}$"
    elseif name ∈ ["domaingap_Inf1_C_high", "domaingap_plus_Inf1_C_high"]
        L"$\mathrm{certificate}(\infty,1)_{\mathbf{u}}^{high}$"

    elseif name ∈ ["domaingap_22_A_low", "domaingap_plus_22_A_low"]
        L"$\mathrm{certificate}(2,2)_{\mathbf{p}_{\mathcal{T}}}^{low}$"
    elseif name ∈ ["domaingap_22_A_high", "domaingap_plus_22_A_high"]
        L"$\mathrm{certificate}(2,2)_{\mathbf{p}_{\mathcal{T}}}^{high}$"
    elseif name ∈ ["domaingap_22_B_low", "domaingap_plus_22_B_low"]
        L"$\mathrm{certificate}(2,2)_{\mathbf{o}}^{low}$"
    elseif name ∈ ["domaingap_22_B_high", "domaingap_plus_22_B_high"]
        L"$\mathrm{certificate}(2,2)_{\mathbf{o}}^{high}$"
    elseif name ∈ ["domaingap_22_C_low", "domaingap_plus_22_C_low"]
        L"$\mathrm{certificate}(2,2)_{\mathbf{u}}^{low}$"
    elseif name ∈ ["domaingap_22_C_high", "domaingap_plus_22_C_high"]
        L"$\mathrm{certificate}(2,2)_{\mathbf{u}}^{high}$"

    elseif name == "binary_certificate_A_low"
        L"$\mathrm{certificate}_{\mathbf{p}_{\mathcal{T}}}^{low}$"
    elseif name == "binary_certificate_A_high"
        L"$\mathrm{certificate}_{\mathbf{p}_{\mathcal{T}}}^{high}$"
    elseif name == "binary_certificate_B_low"
        L"$\mathrm{certificate}_{\mathbf{o}}^{low}$"
    elseif name == "binary_certificate_B_high"
        L"$\mathrm{certificate}_{\mathbf{o}}^{high}$"
    elseif name == "binary_certificate_C_low"
        L"$\mathrm{certificate}_{\mathbf{u}}^{low}$"
    elseif name == "binary_certificate_C_high"
        L"$\mathrm{certificate}_{\mathbf{u}}^{high}$"
    else
        name
    end
end


