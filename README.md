# ns-rom-comparison

Comparison of Reduced Order Methods for the parametric Navier-Stokes equations
on the unit square domain Ω = (0,1)², with parameters μ = (μ₀, μ₁) ∈ [0.1,10] × [1.0,3.0].

## Problem

Given μ = (μ₀, μ₁), find u(μ) such that:

    -μ₀∇·(∇u) + (∇u)u + ∇p = f(x; μ₁)    in Ω
    ∇·u = 0                                  in Ω
    u = 0                                    on ∂Ω

where μ₀ is the kinematic viscosity and f(x; μ₁) is an explicit parametric forcing term.

## Methods

| Method | Description |
|---|---|
| FOM | Full Order Model — Taylor-Hood finite elements, Newton solver |
| POD-Galerkin | Proper Orthogonal Decomposition + reduced Newton |
| POD-NN | POD coefficients learned by a feedforward neural network |
| PINN | Physics-Informed Neural Network |

## Repository Structure
