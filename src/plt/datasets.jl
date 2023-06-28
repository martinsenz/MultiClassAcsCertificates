
using MultiClassAcsCertificates.Data


function _write_datasets_table(datasetnames, caption; suffix="", savepath="res/plots/certification/tex/datasets.tex", standalone=true)

    open(savepath, "w") do io
        if standalone
            _write_header(io)
            println(io, "\\begin{document}")
        end
        println(io, "\\begin{table}")
        println(io, "\\centering")
        println(io, "\\small")
        println(io, "\\begin{tabular}{llll}")
        println(io, "\\toprule")
        println(io, "\\textit{dataset} & \\textit{dim. features} & \\textit{num. samples} & \$\\mathbf{p}_{\\mathcal{S}}\$ \\\\")
        println(io, "\\midrule")
        for name in datasetnames
            @info "dataset = $(name * suffix)"
            d = Data.dataset(name * suffix)
            pyS = round.(Data.class_proportion(d.y_data, Data.classes(d)), digits=2)
            num_samples, num_features = size(d.X_data)
            println(io, "$(name) & $(num_features) & $(num_samples) & $(pyS) \\\\")
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