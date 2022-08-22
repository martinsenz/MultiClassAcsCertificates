"""
    tightness(standalone, full_table)

Generate LaTeX tables from the results of the tightness experiment.
"""
function tightness(standalone::Bool=true, full_table::Bool=true; df_path::String="res/experiments/tightness.csv", save_path::String="res/plots/")
    df = CSV.read(df_path, DataFrame)
    df["cert_bound"] = df["emp_loss_val"] .+ df["ϵ_val"] .+ df["ϵ_cert"] # upper bounded L_S + ϵ_cert
    df["tst_bound"] = df["emp_loss_tst"] .+ df["ϵ_tst"] # upper bounded L_T; 
    df["dist_cert_tst_bound"] = abs.(df["tst_bound"] .- df["cert_bound"]) # absolute deviation between L_S(h) + ϵ and L_T(h)
    df["dist_cert_tst"] = df["emp_loss_tst"] .- df["cert_bound"] # difference 
    dataset_names = []
    scale = 100 # Scaling used for trenary plots. Should not be changed.

    # write Table 2 
    _write_tightness_table(_tightness_table(df; full_table=full_table), save_path * "tex_tables/"; standalone=standalone, full_table=full_table)
    @info "Generate treanry plots..."
    for df_trial in groupby(df, ["dataset", "method", "clf", "delta", "weight", "loss"])

        ℓNormBounded_agg = combine(df_trial, "ℓNormBounded" => mean)
        method_name = df_trial["method"][1]
        ℓNormBounded = ℓNormBounded_agg[1][1]
        pY_S = eval(Meta.parse(df_trial["pY_S"][1]))
        variant_plus = occursin("Plus", method_name) ? true : false

        gap = nothing
        gap(x) = Certification.domaingap_error(method_name, ℓNormBounded, pY_S, x; variant_plus=variant_plus)
        df_agg = combine(groupby(df_trial, "pY_T"), "dist_cert_tst_bound" => mean, "dist_cert_tst" => mean, "cert_bound" => mean, 
                                                    "tst_bound" => mean, "ϵ_cert" => mean)
        pY_T_points = map(i -> tuple(Int.(round.(eval(Meta.parse(df_agg["pY_T"][i])) .* scale))...), 1:nrow(df_agg))
        pY_S_points = map(i -> tuple(Int.(round.(eval(Meta.parse(df_trial["pY_S"][i])) .* scale))...), 1:nrow(df_agg))
        interpolator_gap = Util.NearestNDInterpolator(pY_T_points, df_agg["ϵ_cert_mean"])
        interpolate_gap((x,y,z)) = interpolator_gap(x,y,z)
        interpolator_dist = Util.NearestNDInterpolator(pY_T_points, df_agg["cert_bound_mean"])
        interpolate_dist((x,y,z)) = interpolator_dist(x,y,z)

        # produce identifiable filenames
        clf_name = ""
        if df_trial["clf"][1] == "sklearn.tree.DecisionTreeClassifier"
            clf_name = "DT"
        end
        if df_trial["clf"][1] == "sklearn.linear_model.LogisticRegression"
            clf_name = "LogReg"
        end
        loss = df_trial["loss"][1]
        weight = df_trial["weight"][1]
        delta = replace(string(df_trial["delta"][1]), "." => ",")
        dataset = df_trial["dataset"][1]
        name = "$(method_name)_$(clf_name)_$(loss)_$(weight)_$(delta)_$(dataset)"
        
        # create and save treanry plots
        if !ispath("$(save_path)trenary/$(df_trial["dataset"][1])")
            mkdir("$(save_path)trenary/$(df_trial["dataset"][1])")
        end
        push!(dataset_names, df_trial["dataset"][1])
        _trenary_plot_acc(pY_T_points, pY_S_points, interpolate_dist, "$(save_path)trenary/$(dataset)/$(name)_acc.png"; scale=scale)
        _trenary_plot_predicted_gap(gap, pY_T_points, pY_S_points, interpolate_gap, "$(save_path)trenary/$(dataset)/$(name)_gap.png"; vmin=0.0, scale=scale)
    end
    # generate latex pdf 
    _print_trenary_plots(dataset_names; standalone=standalone)

end

function _tightness_table(df; full_table=true, n_digits=4)
    q_1(x) = quantile(x, 0.25)
    q_2(x) = quantile(x, 0.5)
    q_3(x) = quantile(x, 0.75)
    gdf = if full_table
            groupby(df, ["dataset", "method", "clf", "delta", "weight", "loss"])
        else
            groupby(df, ["dataset", "method", "delta", "weight", "loss"])
        end 
    res = combine(gdf, "dist_cert_tst_bound" => mean, "dist_cert_tst_bound" => std, "dist_cert_tst_bound" => q_1, "dist_cert_tst_bound" => q_2, "dist_cert_tst_bound" => q_3)
    res["dist_cert_tst_bound_mean"] = round.(res["dist_cert_tst_bound_mean"]; digits=n_digits)
    res["dist_cert_tst_bound_std"] = round.(res["dist_cert_tst_bound_std"]; digits=n_digits)
    res["dist_cert_tst_bound_q_1"] = round.(res["dist_cert_tst_bound_q_1"]; digits=n_digits)
    res["dist_cert_tst_bound_q_2"] = round.(res["dist_cert_tst_bound_q_2"]; digits=n_digits)
    res["dist_cert_tst_bound_q_3"] = round.(res["dist_cert_tst_bound_q_3"]; digits=n_digits)
    res
end

function _write_tightness_table(df, save_path; standalone=true, full_table=true)
    open(save_path * "absolute_deviation.tex", "w") do io
        if standalone
            _write_header(io)
            println(io, "\\begin{document}")
        end
        gdf_keys = begin 
            if full_table 
                ["clf", "delta", "weight", "loss"]
            else
                ["delta", "weight", "loss"]
            end        
        end
        for gdf in groupby(df, gdf_keys)
            cap = ""
            if full_table
                clf_name = gdf["clf"][1]
                clf_name = clf_name[findlast(".", clf_name)[1]+1:end]
                delta = gdf["delta"][1]
                loss = gdf["loss"][1]
                weight = gdf["weight"][1]
                cap = "\\caption{MAD and quartiles of the absolute difference between \$\\hat{L}_{S} + \\epsilon\$ and \$\\hat{L}_{\\mathcal{T}} + \\epsilon_{\\mathcal{T}}\$
                       ($(clf_name), $(loss) (weight=$(weight)) and \$\\delta=$(delta)\$) }"
            else
                cap = "\\caption{MAD and quartiles of the absolute difference between \$\\hat{L}_{S} + \\epsilon\$ and \$\\hat{L}_{\\mathcal{T}} + \\epsilon_{\\mathcal{T}}\$.}"
            end
            println(io, "\\begin{table}[!p]")
            println(io, "\\center")
            println(io, cap)
            println(io, "\\small")
            println(io, "\\begin{tabular}{llllll}")
            println(io, "\\toprule")
            println(io, "data set & method & MAD & \$Q_{1}\$ & \$Q_{2}\$ & \$Q_{3}\$ \\\\")
            println(io, "\\midrule")
            for (i,row) in enumerate(eachrow(gdf))
                appendix = mod(i, length(df["dataset"]) / length(unique(df["dataset"]))) == 0 ? "[.5em]" : ""
                println(io, "$(replace(row["dataset"], "_" => " ")) & $(_strategy_names(row["method"])) & 
                                                    \$$(row["dist_cert_tst_bound_mean"]) \\pm $(row["dist_cert_tst_bound_mean"])\$ &
                                                    \$$(row["dist_cert_tst_bound_q_1"])\$ & 
                                                    \$$(row["dist_cert_tst_bound_q_2"])\$ & 
                                                    \$$(row["dist_cert_tst_bound_q_3"])\$\\\\$(appendix)"   )
            end
            println(io, "\\bottomrule")
            println(io, "\\end{tabular}")
            println(io, "\\end{table}")
        end
        if standalone
            println(io, "\\end{document}")
        end
    end
    @info "Written tables to $(save_path)absolute_deviation.tex"
end

function _trenary_plot_acc(points_pY, pY_S_points, interpolate_function, savepath; scale=100)
    d = Dict()
    for (i,j,k) in Util.simplex_iterator(scale)
        push!(d, (i,j) => interpolate_function((i,j,k)))
    end
    figure, tax = Util.ternary.figure(scale=scale)
    figure.set_size_inches(10, 10)
    tax.boundary(linewidth=3.0)
    tax.gridlines(multiple=scale/10, color="white")
    fontsize = 17
    cb_kwargs = Dict(Dict(:use_gridspec => false, :location => "bottom", :pad => -0.03))

    tax.heatmap(d, scale=scale; vmin=0.0, cb_kwargs)
    tax.scatter(pY_S_points, marker="D", label=L"$\mathbf{p}_{S}$", color="red", zorder=5)  
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
    figure.clear()
    tax.close()
    tax = nothing
    figure = nothing
    
end

function _trenary_plot_predicted_gap(gap, points_pY, pY_S_points, interpolate_function, savepath; scale=100, vmin=0.0)
    figure, tax = nothing, nothing
    if gap !== nothing
        figure, tax = Plots.contours_coordinates(gap, collect(0.05:0.05:10), 100)
    else
        figure, tax = Util.ternary.figure(scale=scale)
    end
    figure.set_size_inches(10, 10)
    tax.boundary(linewidth=3.0)
    tax.gridlines(multiple=scale/10, color="white")

    scale = 100
    d = Dict()
    for (i,j,k) in Util.simplex_iterator(scale)
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
    figure.clear()
    tax.close()
    tax = nothing
    figure = nothing 
end

function _caption_from_filename(filename)
    f = split(filename, "_")
    method_name = _strategy_names(f[1] * "_" * f[2])
    clf_name = f[3]
    if clf_name == "LogReg"
        clf_name = "Logistic Regression"
    elseif clf_name == "DT"
        clf_name == "Decision Tree Classifier"
    else
        @error "Classifier name $(clf_name) not recognized!"
    end
    loss = f[4]
    weight = f[5]
    delta = f[6]
    dataset = f[7]
    "\\textbf{$(dataset)}: $(method_name) with $(clf_name), $(loss) (weight=$(weight)) and \$\\delta=$(delta)\$." 
end

function _print_trenary_plots(dataset_names; standalone=true, load_dir::String="res/plots/trenary/")
    open(load_dir * "tightness_plots.tex", "w") do io 
        if standalone
            _write_header(io)
            println(io, "\\begin{document}")
        end
        for dataset in Base.Filesystem.readdir(load_dir)
            if dataset ∈ dataset_names
                files = Base.Filesystem.readdir(load_dir * dataset * "/")
                for file in unique(map(x -> chop(x; tail=8), files))
                    println(io, "\\begin{figure}[!p]")
                    println(io, "\\centering")
                    println(io, "\\begin{subfigure}[b]{.49\\textwidth}")
                    println(io, "\\centering")
                    println(io, "\\includegraphics[width=\\textwidth]{$(dataset * "/" * file)_gap}")
                    println(io, "\\caption*{Certified domain-induced error \$\\epsilon\$}")
                    println(io, "\\end{subfigure}")
                    println(io, "\\hfill")
                    println(io, "\\begin{subfigure}[b]{.49\\textwidth}")
                    println(io, "\\centering")
                    println(io, "\\includegraphics[width=\\textwidth]{$(dataset * "/" * file)_acc}")
                    println(io, "\\caption*{\$ |\\hat{L}_{S} + \\epsilon - \\hat{L}_{\\mathcal{T}} + \\epsilon_{\\mathcal{T}}| \$}")
                    println(io, "\\end{subfigure}")
                    println(io, "\\hfill")
                    println(io, "\\caption{ $(_caption_from_filename(file)) }" )
                    println(io, "\\end{figure}")
                end
            end
        end
        if standalone
            println(io, "\\end{document}")
        end
        @info "Written trenary plots to $(load_dir)tightness_plots.tex"
    end
end