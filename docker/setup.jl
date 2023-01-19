using Pkg
@info "Installing package dependencies with setup.jl"
pkg"instantiate"
pkg"build"
pkg"precompile"

using Conda

# install Python dependencies
Conda.runconda(`install -y scikit-learn`)
Conda.runconda(`install -y pandas`)
Conda.runconda(`install -y python-ternary`)
#Conda.pip_interop(true)
#Conda.pip("install", "git+https://github.com/scikit-activeml/scikit-activeml.git@unstable")
