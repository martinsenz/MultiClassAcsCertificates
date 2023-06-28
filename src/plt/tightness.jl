const scale = 100 # scaling factor for ternary plots

"""
    tightness(df_path::String)

Generate LaTeX tables from the results of the tightness experiment.
"""
function tightness(df_path::String="res/experiments/tightness.csv")
    df = CSV.read(df_path, DataFrame)
    df[!, "pY_S"] = eval.(Meta.parse.(df[!, "pY_S"]))
    df[!, "pY_T"] = eval.(Meta.parse.(df[!, "pY_T"]))
    df = df[sortperm(df[!, "pY_T"]), :]
    n_classes = length(df[!, "pY_S"][1])
    filename = split(last(split(df_path, "/")), ".")[1]

    df[!, "certification_bound"] = df[!, "emp_loss_val"] .+ df[!, "ϵ_val"] .+ df[!, "ϵ_cert"]
    df[!, "test_bound"] = df[!, "emp_loss_tst"] .+ df[!, "ϵ_tst"]
    df[!, "failures"] = df[!, "certification_bound"] .< df[!, "emp_loss_tst"]
    df[!, "distance_certification_tst_bound"] = abs.(df[!, "certification_bound"] .- df[!, "test_bound"])
    if n_classes == 2 
        _correctness_table(df, "res/plots/certification/tex/" * "CorrectnessFullTable" * filename * ".tex")
        _correctness_table(df, "res/plots/certification/tex/" * "Correctness" * filename * ".tex"; full_table=false)
        _tightness_table(df, "res/plots/certification/tex/" * "madFullTable" * filename * ".tex")
        _tightness_table(df, "res/plots/certification/tex/" * "madTable" * filename * ".tex"; full_table=false)
        _binary_tightness_plots(df, "res/plots/certification/tex/" * filename * ".tex")
    elseif n_classes == 3
        _correctness_table(df, "res/plots/certification/tex/" * "CorrectnessFullTable" * filename * ".tex")
        _correctness_table(df, "res/plots/certification/tex/" * "Correctness" * filename * ".tex"; full_table=false)
        _tightness_table(df, "res/plots/certification/tex/" * "tightnessFullTable" * filename * ".tex")
        _tightness_table(df, "res/plots/certification/tex/" * "tightnessTable" * filename * ".tex"; full_table=false)
        _ternary_plots(df, "$(filename).tex")
    else
        @info "The evaluation can currently be performed only under two or three classes!"
    end
end

function _correctness_table(df::DataFrame, savepath::String; full_table=true, standalone=true)

    gid = ["clf", "delta", "weight", "loss", "method"]
    df_agg = begin
        if full_table # additional groupby datasets 
            combine(groupby(df, vcat(gid, "dataset")), "failures" => sum,"failures" => mean)
        else
            combine(groupby(df, gid), "failures" => sum,"failures" => mean)
        end
    end

    return df_agg
    
    # write to latex table
    classifiername = _classifier_names(df[!, "clf"][1])
    loss = df[!,"loss"][1]
    delta = df[!, "delta"][1]
    n_datasets = length(unique(df[!, "dataset"]))
    n_methods = length(unique(df[!, "method"]))
    open(savepath, "w") do io
        if standalone
            _write_header(io)
            println(io, "\\begin{document}")
        end
        caption = "Correctness of the certificates with clf=$(classifiername), loss=$(loss) and \$\\delta=$(delta)\$"
        println(io, "\\begin{table}")
        println(io, "\\centering")
        println(io, "\\small")
        if full_table
            println(io, "\\begin{tabular}{llcl}")
        else
            println(io, "\\begin{tabular}{lcl}")
        end
        println(io, "\\toprule")
        if full_table
            println(io, "\\textit{dataset} & \\textit{method} & \\textit{failures} & \\textit{fraction of failures}\\\\")
        else
            println(io, "\\textit{method} & \\textit{failures} & \\textit{fraction of failures}\\\\")
        end
        println(io, "\\midrule")
        if full_table
            for (i, datasetname) in enumerate(unique(df_agg[!,"dataset"]))
                for (j, methodname) in enumerate(unique(df_agg[!, "method"]))
                    idx = (df_agg[!, "dataset"] .== datasetname) .& (df_agg[!, "method"] .== methodname)
                    row = df_agg[idx, :]
                    row_print = "$(replace(datasetname, "_" => "")) & $(_strategy_names(methodname)) & $(row[!, "failures_sum"][1]) & $(row[!, "failures_mean"][1])\\\\"
                    if (j == n_methods) && (i != n_datasets)
                        row_print *= "[3pt]"
                    end
                    println(io, row_print)
                end
            end
        else
            for methodname in unique(df_agg[!, "method"])
                row = df_agg[df_agg[!, "method"] .== methodname, :]
                println(io, "$(_strategy_names(methodname)) & $(row[!, "failures_sum"][1]) & $(row[!, "failures_mean"][1])\\\\")
            end
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\caption{")
        println(io, caption)
        println(io, "}")
        println(io, "\\end{table}")
        if standalone
            println(io, "\\end{document}")
        end
        close(io)
    end
    @info "Written table to $(savepath)"
end

function _binary_tightness_plots(df::DataFrame, savepath::String)

    # set up TikzDocument with one page per group plot
    resetPGFPlotsPreamble()
    pushPGFPlotsPreamble("\\usepackage{amsmath,amssymb,booktabs,hyperref}")
    pushPGFPlotsPreamble("\\usepackage[paperheight=14in, paperwidth=7in, margin=1in]{geometry}")
    pushPGFPlotsPreamble("\\usetikzlibrary{calc}")
    pushPGFPlotsPreamble("\\usepgfplotslibrary{hvlines}")
    pushPGFPlotsPreamble("\\definecolor{tu01}{HTML}{84B818}")
    pushPGFPlotsPreamble("\\definecolor{tu02}{HTML}{D18B12}")
    pushPGFPlotsPreamble("\\definecolor{tu03}{HTML}{1BB5B5}")
    pushPGFPlotsPreamble("\\definecolor{tu04}{HTML}{F85A3E}")
    pushPGFPlotsPreamble("\\definecolor{tu05}{HTML}{4B6CFC}")

    # one page per combination of loss and delta
    colors = sort(collect(keys(tu_colors)))
    document = TikzDocument()
    inspect = nothing
    for gdf in groupby(df, ["dataset", "clf", "delta", "weight", "loss"])
        domain = unique(vcat([gdf[:, "pY_T"][x][2] for x in 1:size(gdf, 1)]...))
        pY = gdf[!, "pY_S"][1][2]
        agg = combine(groupby(gdf, ["method", "pY_T"]), 
                            "emp_loss_val" => mean,
                            "emp_loss_tst" => mean,
                            "ϵ_cert" => mean,
                            "ϵ_val" => mean,
                            "ϵ_tst" => mean)
        agg_norm = combine(groupby(agg, "pY_T"), 
                            "emp_loss_val_mean" => mean, 
                            "emp_loss_tst_mean" => mean, 
                            "ϵ_val_mean" => mean, 
                            "ϵ_tst_mean" => mean)
        inspect = agg, agg_norm
        #break
        axis = Axis(; style = join([
            "title={}",
            "title style={font=\\scriptsize, text width=.2\\linewidth, align=center}",
            "xlabel={\$p_\\mathcal{T}\$}",
            "ylabel={\$L_\\mathcal{T}(h)\$}",
            "xlabel style={font=\\footnotesize}",
            "ylabel style={font=\\footnotesize}",
            "xmode=log",
            "scale=1.",
            "vertical line={at=$(pY), style={gray, very thin}}",
            "legend cell align={left}",
            "legend style={draw=none,fill=none,at={(1.1,.5)},anchor=west,row sep=.25em}",
            "yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2}",
            "scaled y ticks=false"
        ], ", "))
        push!(axis, PGFPlots.Plots.Linear(
            domain,
            agg_norm[!, "emp_loss_val_mean_mean"];
            style = "semithick, densely dotted, gray, mark=*, mark options={scale=.8, solid, fill=gray}",
            legendentry="LT"
        )) # L_T
        target_domain_bound = agg_norm[!, "emp_loss_val_mean_mean"] .+ agg_norm[!, "ϵ_tst_mean_mean"]
        push!(axis, PGFPlots.Plots.Linear(
            domain,
            target_domain_bound;
            style = "semithick, dashed, blue, mark=triangle*, mark options={scale=1, solid, fill=blue}",
            legendentry="LT Bound"
        )) # L_T + eps_T
        method_id = unique(agg[!, "method"])
        for (i, agg) in enumerate(groupby(agg, "method"))
            certification_bound = agg[!, "emp_loss_val_mean"] .+ agg[!, "ϵ_val_mean"] .+ agg[!, "ϵ_cert_mean"]
            push!(axis, PGFPlots.Plots.Linear(
                domain,
                certification_bound;
                style = "semithick, solid, $(colors[i]), mark=square*, mark options={scale=.8, solid, fill=$(colors[i])}",
                legendentry="$(_strategy_names(method_id[i]))"
            )) # certificates
        end
        push!(
            document,
            PGFPlots.plot(axis);
            caption = "$(replace(gdf[!,"dataset"][1], "_"=>"")): $(gdf[!,"loss"][1]), $(_classifier_names(gdf[!,"clf"][1])) with \$\\delta = $(gdf[!,"delta"][1])\$"
        ) 
    end
    #return inspect
    save(TEX(savepath), document) # .tex export
    @info "Plotting to $(savepath)"
end

function _tightness_table(df::DataFrame, savepath::String; full_table=true, standalone=true, digits=4)
    q1(x) = quantile(x, 0.25)
    q2(x) = quantile(x, 0.5)
    q3(x) = quantile(x, 0.75)
    gdf = begin
        if full_table
            groupby(df, ["dataset", "method", "clf", "delta", "weight", "loss"])
        else
            groupby(df, ["method", "clf", "delta", "weight", "loss"])
        end
    end
    df_agg = combine(gdf, "distance_certification_tst_bound" .=> [mean, std, q1, q2, q3])
    classifiername = _classifier_names(df[!, "clf"][1])
    loss = df[!,"loss"][1]
    delta = df[!, "delta"][1]
    n_datasets = length(unique(df[!, "dataset"]))
    n_methods = length(unique(df[!, "method"]))
    caption = "MAD and quartiles of the absolute difference between \$\\hat{L}_{S} + \\epsilon\$ and \$\\hat{L}_{\\mathcal{T}} + \\epsilon_{\\mathcal{T}}\$ ($(classifiername), $(loss) and \$\\delta=$(delta)\$)"
    open(savepath, "w") do io
        if standalone
            _write_header(io)
            println(io, "\\begin{document}")
        end
        println(io, "\\begin{table}[h]")
        println(io, "\\center")
        println(io, "\\small")
        println(io, "\\caption{")
        println(io, caption)
        println(io, "}")
        topruleprint = "method & MAD & \$Q_{1}\$ & \$Q_{2}\$ & \$Q_{3}\$ \\\\"
        if full_table
            println(io, "\\begin{tabular}{llcccc}")
            topruleprint = "dataset & " * topruleprint
        else
            println(io, "\\begin{tabular}{lcccc}")
        end
        println(io, "\\toprule")
        println(io, topruleprint)
        println(io, "\\midrule")
        if full_table
            for (i, datasetname) in enumerate(unique(df_agg[!,"dataset"]))
                for (j, methodname) in enumerate(unique(df_agg[!, "method"]))
                    idx = (df_agg[!, "dataset"] .== datasetname) .& (df_agg[!, "method"] .== methodname)
                    row = df_agg[idx, :]
                    _mean = round(row[!,"distance_certification_tst_bound_mean"][1]; digits=digits)
                    _std = round(row[!,"distance_certification_tst_bound_std"][1]; digits=digits)
                    _q1 = round(row[!,"distance_certification_tst_bound_q1"][1]; digits=digits)
                    _q2 = round(row[!,"distance_certification_tst_bound_q2"][1]; digits=digits)
                    _q3 = round(row[!,"distance_certification_tst_bound_q3"][1]; digits=digits)
                    row_print = "$(replace(datasetname, "_" => "")) & $(_strategy_names(methodname)) & \$$(_mean) \\pm $(_std)\$ & \$$(_q1)\$ & \$$(_q2)\$ & \$$(_q3)\$ \\\\"
                    if (j == n_methods) && (i != n_datasets)
                        row_print *= "[3pt]"
                    end
                    println(io, row_print)
                end
            end
        else
            for (j, methodname) in enumerate(unique(df_agg[!, "method"]))
                idx = df_agg[!, "method"] .== methodname
                row = df_agg[idx, :]
                _mean = round(row[!,"distance_certification_tst_bound_mean"][1]; digits=digits)
                _std = round(row[!,"distance_certification_tst_bound_std"][1]; digits=digits)
                _q1 = round(row[!,"distance_certification_tst_bound_q1"][1]; digits=digits)
                _q2 = round(row[!,"distance_certification_tst_bound_q2"][1]; digits=digits)
                _q3 = round(row[!,"distance_certification_tst_bound_q3"][1]; digits=digits)
                row_print = "$(_strategy_names(methodname)) & \$$(_mean) \\pm $(_std)\$ & \$$(_q1)\$ & \$$(_q2)\$ & \$$(_q3)\$ \\\\"
                if (j == n_methods)
                    row_print *= "[3pt]"
                end
                println(io, row_print)
            end
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}")
        if standalone
            println(io, "\\end{document}")
        end
    end
    @info "Written table to $(savepath)"
    
end

function _ternary_plot_class_prior_distribution(class_prior_distribution, pY_T, savepath; scale=100, n_points=10000, legend=true)

    figure, tax = Plots.ternary.figure(scale=scale)
    #figure.suptitle(L"$\mathbb{E}$ = " * "$(mean(class_prior_distribution))", fontsize=20)
    figure.set_size_inches(10, 10)
    tax.boundary(linewidth=3.0)
    tax.gridlines(multiple=scale/10, color="white")
    fontsize = 17
    cb_kwargs = Dict(Dict(:use_gridspec => false, :location => "bottom", :pad => -0.03))

    f(x) = Distributions.pdf(class_prior_distribution, x)
    tax.heatmapf(f, scale=scale, colorbar=false; cb_kwargs)
    #tax.contourf(f, scale=scale, colorbar=true; cb_kwargs)
    tax.scatter(pY_T, marker="D", label=L"$\mathbf{p}_{\mathcal{T}}$", color="red", zorder=10)  
    tax.left_axis_label("C3", fontsize=fontsize, position=(-0.10,0.3), rotation=0.0)
    tax.right_axis_label("C2", fontsize=fontsize, position=(0.17,0.96), rotation=0.0)
    tax.bottom_axis_label("C1", fontsize=fontsize, position=(0.79,0.05))
    ticks_labels = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    tax.ticks(ticks=ticks_labels)
    if legend
        tax.legend(loc="upper right", fontsize=20)
    end
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    tax.savefig(savepath, transparent=true, pad_inches=0.0, bbox_inches="tight")
    figure.clear()
    tax.close()
    tax = nothing
    figure = nothing
    
end

function _ternary_plots(df::DataFrame, savename::String)

    for trial in groupby(df, ["dataset", "method", "clf", "delta", "weight", "loss"])
    
        agg = combine(groupby(trial, "pY_T"), "ϵ_cert" => mean, "distance_certification_tst_bound" => mean)

        method = trial[!, "method"][1]
        variant_plus = occursin("Plus", method) ? true : false
        pY_S = mean(trial[!, "pY_S"])
        pY_S_point = [MultiClassAcsCertificates.Plots._to_point(pY_S)]
        pY_T_points = map(x -> MultiClassAcsCertificates.Plots._to_point(x), agg[!, "pY_T"])
        ℓNormBounded = mean(trial[!, "ℓNormBounded"])

        gap(x) = Certification.domaingap_error(method, ℓNormBounded, pY_S, x; variant_plus=variant_plus)
        interpolator_gap = MultiClassAcsCertificates.Plots.NearestNDInterpolator(pY_T_points, agg[!, "ϵ_cert_mean"])
        interpolator_distance = MultiClassAcsCertificates.Plots.NearestNDInterpolator(pY_T_points, agg[!,"distance_certification_tst_bound_mean"])
        interpolate_distance((x,y,z)) = interpolator_distance(x,y,z)

        # produce identifiable filenames
        clf_id = begin 
            if trial[!, "clf"][1] == "sklearn.tree.DecisionTreeClassifier"
                    "DT"
            elseif trial[!, "clf"][1] == "sklearn.linear_model.LogisticRegression"
                    "LogReg"
            elseif trial[!, "clf"][1] == "sklearn.neural_network.MLPClassifier"
                    "MLP"
            else
                    ""
            end
        end
        loss = trial[!,"loss"][1]
        weight = trial[!,"weight"][1]
        delta = replace(string(trial[!,"delta"][1]), "." => ",")
        dataset = replace(trial[!,"dataset"][1], "_" => "")
        file_id = "$(method)_$(clf_id)_$(loss)_$(weight)_$(delta)_$(dataset)"
        savepath="res/plots/certification/ternary/$(file_id)"
        MultiClassAcsCertificates.Plots._ternary_predicted_domaingap(gap, pY_S_point, interpolator_gap, savepath*"_gap.png")
        MultiClassAcsCertificates.Plots._ternary_distance(pY_S_point, interpolate_distance, savepath*"_mad.png")

    end
    _print_ternary_plots()

end

function _ternary_distance(pY_S_point, interpolate_function, savepath)
    d = Dict()
    for (i,j,k) in Plots.simplex_iterator(scale)
        push!(d, (i,j) => interpolate_function((i,j,k)))
    end
    figure, tax = Plots.ternary.figure(scale=scale)
    figure.set_size_inches(10, 10)
    tax.boundary(linewidth=3.0)
    tax.gridlines(multiple=scale/10, color="white")
    fontsize = 17
    cb_kwargs = Dict(Dict(:use_gridspec => false, :location => "bottom", :pad => -0.03))

    tax.heatmap(d, scale=scale; vmin=0.0, cb_kwargs)
    tax.scatter(pY_S_point, marker="D", label=L"$\mathbf{p}_{S}$", color="red", zorder=5)  
    tax.left_axis_label("C3", fontsize=fontsize, position=(-0.10,0.3), rotation=0.0)
    tax.right_axis_label("C2", fontsize=fontsize, position=(0.17,0.96), rotation=0.0)
    tax.bottom_axis_label("C1", fontsize=fontsize, position=(0.79,0.05))
    ticks_labels = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    tax.ticks(ticks=ticks_labels)
    #tax.legend(loc="upper right", fontsize=20)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    d = nothing
    tax.savefig(savepath, transparent=true, pad_inches=0.0, bbox_inches="tight")
    @info "Written ternary plot $(savepath)"
    figure.clear()
    tax.close()
    tax = nothing
    figure = nothing
end

function _ternary_predicted_domaingap(gap, pY_S_points, interpolate_function, savepath; vmin=0.0)

    figure, tax = nothing, nothing
    if gap !== nothing
        figure, tax = Plots.contours_coordinates(gap, collect(0.05:0.05:10), 100)
    else
        figure, tax = Plots.ternary.figure(scale=scale)
    end
    figure.set_size_inches(10, 10)
    tax.boundary(linewidth=3.0)
    tax.gridlines(multiple=scale/10, color="white")

    d = Dict()
    for (i,j,k) in Plots.simplex_iterator(scale)
        push!(d, (i,j) => interpolate_function((i,j,k)))
    end
    tax.scatter(pY_S_points, marker="D", label=L"$\mathbf{p}_{S}$", color="red", zorder=5)  
    fontsize = 17
    cb_kwargs = Dict(Dict(:use_gridspec => false, :location => "bottom", :pad => -0.03))
    tax.heatmap(d, scale=scale; cmap="jet", vmin=vmin, cb_kwargs)

    tax.left_axis_label("C3", fontsize=fontsize, position=(-0.10,0.3), rotation=0.0)
    tax.right_axis_label("C2", fontsize=fontsize, position=(0.17,0.96), rotation=0.0)
    tax.bottom_axis_label("C1", fontsize=fontsize, position=(0.79,0.05))
    ticks_labels = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    tax.ticks(ticks=ticks_labels)
    #tax.legend(loc="upper right")
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")

    d = nothing
    tax.savefig(savepath, transparent=true, pad_inches=0.0, bbox_inches="tight")
    @info "Written ternary plot $(savepath)"
    figure.clear()
    tax.close()
    tax = nothing
    figure = nothing 
end

function _caption_from_filename(filename)
    f = split(filename, "_")
    method_name = _strategy_names(f[1] * "_" * f[2] * "_" * f[3])
    clf_name = f[4]
    if clf_name == "LogReg"
        clf_name = "Logistic Regression"
    elseif clf_name == "DT"
        clf_name = "Decision Tree Classifier"
    elseif clf_name == "MLP"
        clf_name = "MLPClassifier"
    else
        @error "Classifier name $(clf_name) not recognized!"
    end
    loss = f[5]
    weight = f[6]
    delta = f[7]
    dataset = f[8]
    "\\textbf{$(dataset)}: $(method_name) with $(clf_name), $(loss) (weight=$(weight)) and \$\\delta=$(delta)\$." 
end

function _print_ternary_plots(; standalone=true, load_dir::String="res/plots/certification/ternary/", savepath="res/plots/certification/tex/tightness_ternary_plots.tex")
    open(savepath, "w") do io 
        if standalone
            _write_header(io)
            println(io, "\\begin{document}")
        end
        println(io, "\\graphicspath{ {../ternary/} }")
        files = Base.Filesystem.readdir(load_dir)
        filter!(x -> endswith(x, ".png"), files)
        files = unique(chop.(files; tail=8))

        for filename in files
            println(io, "\\begin{figure}[H]")
            println(io, "\\centering")
            println(io, "\\begin{subfigure}[b]{.49\\textwidth}")
            println(io, "\\centering")
            println(io, "\\includegraphics[width=\\textwidth]{$(filename)_gap}")
            println(io, "\\caption*{Certified domain-induced error \$\\epsilon\$}")
            println(io, "\\end{subfigure}")
            println(io, "\\hfill")
            println(io, "\\begin{subfigure}[b]{.49\\textwidth}")
            println(io, "\\centering")
            println(io, "\\includegraphics[width=\\textwidth]{$(filename)_mad}")
            println(io, "\\caption*{\$ |\\hat{L}_{S} + \\epsilon - \\hat{L}_{\\mathcal{T}} - \\epsilon_{\\mathcal{T}}| \$}")
            println(io, "\\end{subfigure}")
            println(io, "\\hfill")
            println(io, "\\caption{ $(_caption_from_filename(filename)) }" )
            println(io, "\\end{figure}")        
        end
        if standalone
            println(io, "\\end{document}")
        end
        @info "Written ternary plots to $(load_dir)tightness_plots.tex"
    end
end