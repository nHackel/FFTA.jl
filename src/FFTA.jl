module FFTA

using Primes, DocStringExtensions, Reexport, MuladdMacro, LinearAlgebra

import AbstractFFTs: AbstractFFTs, Plan, AbstractFFTBackend

include("callgraph.jl")
include("algos.jl")
include("plan.jl")

end
