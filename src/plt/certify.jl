"""
    certify(df_path, save_path, standalone)

Generate LaTeX tables from the results of the certify experiment.
"""
function certify(df_path::String="res/experiments/certify.csv", save_path::String="res/plots/",standalone::Bool=true)
    df = CSV.read(df_path, DataFrame)
    df["pY_S"] = eval.(Meta.parse.(df["pY_S"]))
    res = combine(groupby(df, ["dataset","method", "clf", "delta" ,"epsilon", "loss", "weight"]), "pY_S" => mean, "L_S" => mean, "delta_p" => mean)
    _write_certify_table(res, save_path * "tex_tables/", standalone)
    @info "Writing tables to $(save_path)tex_tables/certify.tex"
end

# write table 1
function _write_certify_table(df::DataFrame, save_path::String, standalone::Bool)
    open(save_path * "certify.tex", "w") do io 
        if standalone
            _write_header(io)
            println(io, "\\begin{document}")
        end
        for gdf in groupby(df, ["method", "loss", "weight", "delta", "epsilon"])
            delta = gdf["delta"][1]
            epsilon = gdf["epsilon"][1]
            loss = gdf["loss"][1]
            weight = gdf["weight"][1]
            method = _strategy_names(gdf["method"][1])
            println(io, "\\begin{table}[!p]")
            println(io, "\\center")
            println(io, "\\caption{Feasible class proportions \$\\Delta p^{*}\$, according to 
                        $(method) certificates,
                        which are computed for a $(loss) (weight=$(weight)) with \$\\epsilon=$(epsilon)\$ and \$\\delta=$(delta)\$.}")
            println(io, "\\small")
            println(io, "\\begin{tabular}{lllll}")
            println(io, "\\toprule")
            println(io, "data set & classifier & \$L_{S}(h)\$ & \$\\mathbf{p}_{S}^{\\top}\$ & \$\\Delta p^{*}\$ \\\\")
            println(io, "\\midrule")
            for (i, row) in enumerate(eachrow(gdf))
                appendix = mod(i, length(gdf["dataset"]) / length(unique(gdf["dataset"]))) == 0 ? "[.5em]" : ""
                clf_name = row["clf"]
                clf_name = clf_name[findlast(".", clf_name)[1]+1:end]
                println(io, "$(replace(row["dataset"], "_" => " ")) &
                            $(clf_name)  & 
                            \$$(round.(row["L_S_mean"], digits=6))\$  &
                            \$$(round.(row["pY_S_mean"];digits=2))\$  & 
                            \$$(round.(row["delta_p_mean"]; digits=6))\$ \\\\$(appendix)")
            end
            println(io, "\\bottomrule")
            println(io, "\\end{tabular}")
            println(io, "\\end{table}")
        end
        if standalone
            println(io, "\\end{document}")
        end
    end
end