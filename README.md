# Certifiable Active Class Selection in Multi-Class Classification

Supplementary material for our publication.

```bibtex
@InProceedings{senz2022active,
  author    = {Martin Senz and Mirko Bunse and Katharina Morik},
  title     = {Certifiable Active Class Selection in Multi-Class Classification},
  booktitle = {Workshop on Interactive Adaptive Learning},
  year      = {2022},
  note      = {To appear},
  publisher = {{CEUR} Workshop Proceedings}
}
```

This publication is an extension of the [theory](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_598.pdf) appliceable for the certification of binary classifications. The implementation of the certification approach for binary problems can be found [here](https://github.com/mirkobunse/AcsCertificates.jl).


## Reproducing plots

**Preliminaries:** You need to have `julia` and `pdflatex` installed. We conducted the experiments with Julia v1.6 and TexLive on Linux. Sometimes the `Manifest.toml` file causes trouble; since the dependencies are already defined in the `Project.toml`, you should be able to safely delete the manifest file and try again without it.

### Project Setup in a Julia REPL

```julia
using Pkg; Pkg.activate(.) # activate project environment 
using MultiClassAcsCertificates # import module
MultiClassAcsCertificates.run_all_experiments() # this can take some time...
```
The resulting .tex files can be compiled with pdflatex, generating Table 1, Table 2, Fig: 1, Fig: 2. 

### Adapting experiments
The experiments are configured by .yml files and are customizable. You can alter the configurations in the `conf/` directory, e.g. try different seeds, classifiers, or data sets.

More severe adaptations are possible through changing the code. To this end, the Julia modules in the `src/` directory separate concerns in the following ways:

- `MultiClassAcsCertificates` is the top-level module.
- `Certification` implements the method of our multi-class certification approach.
- `Data` downloads and reads the data and provides some data-related utility functions.
- `Experiments` take configuration files as inputs and produce raw results, i.e. they write all evaluations of all trials to an output CSV file.
- `Plots` take the raw results as inputs, aggregate them, and produce TEX files with code for plots.
- `Util` is mainly used to import used Python libraries.

## Using our certificates elsewhere

This project can become a dependency in any other Julia project. Certificates can then be created by simply calling the constructor `MultiClassAcsCertificates(L, y_h, y; kwargs...)` with a loss function `L`, predictions `y_h`, and the validation ground-truths `y`. The keyword arguments provide additional parameters like delta and class weights. Many decomposable loss functions are already available through [the LossFunctions.jl package](https://github.com/JuliaML/LossFunctions.jl).

```julia
# let y_val be validation labels and y_h be corresponding predictions
using MultiClassAcsCertificates.Certification, LossFunctions
c = MultiClassCertificate(ZeroOneLoss(), y_h, y_val)

?MultiClassCertificate # inspect the documentation
```

## Support

If you encounter any problem, please file a GitHub issue.