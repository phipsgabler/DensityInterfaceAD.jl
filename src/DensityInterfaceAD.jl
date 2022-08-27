module DensityInterfaceAD

using DensityInterface
using LogDensityProblems: LogDensityProblems

const LDP = LogDensityProblems

export ADConfig


"""
struct DifferentiableLogFuncDensity{F}

Wrapper type similar to `DensityInterface.LogFuncDensity`, but satisfying both the
`DensityInterface` and the `LogDensityProblems` interface.  This necessitates the specification of
the `dimension` field for the input dimension.

For actual differentiation, use `LogDensityProblems.ADGradient` around this, or go directly through
the [`logfuncdensity`](@ref) method with an [`ADConfig`](@ref) argument.
"""
struct DifferentiableLogFuncDensity{F}
    _log_f::F
    _dimension::Int
end

DifferentiableLogFuncDensity(ℓ, dimension) = _construct_dlfd(DensityKind(ℓ), ℓ, dimension)
function _construct_dlfd(::Type{<:DensityInterface.IsOrHasDensity}, ℓ, dimension)
    log_f = logdensityof(ℓ)
    return DifferentiableLogFuncDensity{typeof(log_f)}(log_f, dimension)
end
function _construct_dlfd(::Type{<:DensityInterface.NoDensity}, ℓ, dimension)
    return DifferentiableLogFuncDensity{typeof(ℓ)}(ℓ, dimension)
end

function Base.show(io::IO, ℓ::DifferentiableLogFuncDensity)
    print(io, nameof(typeof(ℓ)), "(")
    show(io, ℓ._log_f)
    print(io, ", ", ℓ._dimension, ")")
end

LDP.logdensity(ℓ::DifferentiableLogFuncDensity, x) = ℓ._log_f(x)
LDP.dimension(ℓ::DifferentiableLogFuncDensity) = ℓ._dimension
LDP.capabilities(::Type{<:DifferentiableLogFuncDensity}) = LDP.LogDensityOrder{0}()

DensityInterface.logdensityof(ℓ::DifferentiableLogFuncDensity) = ℓ._log_f
DensityInterface.logdensityof(ℓ::DifferentiableLogFuncDensity, x) = ℓ._log_f(x)
DensityInterface.DensityKind(::DifferentiableLogFuncDensity) = DensityInterface.IsDensity()


# TYPE PIRACY -- this should really go into LDP!
DensityInterface.logdensityof(ℓ::LDP.ADGradientWrapper) =
    DifferentiableLogFuncDensity(Base.Fix1(LDP.logdensity, ℓ), LDP.dimension(ℓ))
DensityInterface.logdensityof(ℓ::LDP.ADGradientWrapper, x) = LDP.logdensity(ℓ, x)
DensityInterface.DensityKind(ℓ::LDP.ADGradientWrapper) = DensityInterface.IsDensity()


struct ADConfig{kind, K}
    dimension::Int
    kwargs::K
end

"""
    ADConfig(kind, dimension; kwargs...)

Construct a configuration object for a wrapped differntiable density object with backend `kind`,
input dimension `dimension`, and additional setup `kwargs`.  See [`logfuncdensity`](@ref).
"""
ADConfig(kind::Symbol, dimension::Int, kwargs...) = ADConfig(Val{kind}(), dimension; kwargs...)
ADConfig(::Val{kind}, dimension::Int, kwargs...) where {kind} = ADConfig{kind, typeof(kwargs)}(dimension, kwargs)

"""
    logfuncdensity(log_f, ad::ADConfig)

Construct a differentiable log density object from a log density function.  The necessary dimension
and optional arguments can be specified via the `ADConfig` object:

```julia
logfuncdensity(ϕ, ADConfig(:ReverseDiff, 2; compile=Val(true)))
```

produces an

```julia
LogDensityProblems.ADGradient(:ReverseDiff, ...; compile=Val(true))
```

with `ϕ` wrapped such that its `dimension` is `2`.
"""
function logfuncdensity(log_f, ad::ADConfig{kind}; kwargs...) where {kind}
    return LDP.ADgradient(Val{kind}(), DifferentiableLogFuncDensity(log_f, ad.dimension); kwargs...)
end

end # module DensityInterfaceAD
