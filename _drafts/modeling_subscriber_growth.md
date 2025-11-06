


There are a lot of ways to do this and it depends on what your inputs are. In this post I'll show how to work with subscriber growth as it comes out of Substack, although it should work in many cases.


## The Equation

Let's talk about the equation

$$ \Delta S_t = r \cdot S_{t-1}\left(1 - \frac{S_{t-1}}{K}\right) + g(a_t) + \gamma \cdot \text{pulse}_t + \varepsilon_t $$



$$ \Delta S_t = r_{\mathrm{seg}(t)} S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right) + \gamma_{\mathrm{pulse}}\, \mathrm{pulse}_t + \gamma_{\mathrm{step}}\, \mathrm{step}_t + \gamma_{\mathrm{exog}} x_t + \varepsilon_t $$



St - subscribers at time t

S\_{t-1} - subscribers at time t-1

r - growth rate

K - carrying capacity

g(at) - ad effect

γpulse - pulse effect

εt - error

$$ \varepsilon_t $$ - error

$ \varepsilon_t $ - error

$$ \Delta S_t = r \cdot S_{t-1}\left(1 - \frac{S_{t-1}}{K}\right) + g(a_t) + \gamma \cdot \text{pulse}_t + \varepsilon_t $$

$$ \Delta S_t = \varepsilon_t $$

$$ \varepsilon_t $$

Now, let's look in more detail.


$$ \Delta S_t =
\underbrace{r_{\mathrm{seg}(t)} S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right)}_{\text{endogenous/logistic growth (segment r)}} + \underbrace{\gamma_{\mathrm{pulse}}\, \mathrm{pulse}_t}_{\text{one-month shock}} + \underbrace{\gamma_{\mathrm{step}}\, \mathrm{step}_t}_{\text{persistent additive drift}} + \gamma_{\mathrm{exog}} x_t $$



If I were building a modeler, I would think of doing it in two phases:
1. raw data -> function parameters
2. function parameters -> projections

## Endling Phase 1

OK, before we end phase 1, let's look at what we've done. We've input raw variables and come up with parameters for our equation. Here are all the parameters we should have:

* Carrying capacity

* Growth rates for each segment

* Breakpoint locations

*    "gamma_pulse": 0.0,
    "gamma_step": 0.0,
    "gamma_exog": 0.0,
    "gamma_intercept": 2.9755717864151956

## Starting Phase 2





