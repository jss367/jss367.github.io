


There are a lot of ways to do this and it depends on what your inputs are. In this post I'll show how to work with subscriber growth as it comes out of Substack, although it should work in many cases.


## The Equation

Let's talk about the equation

$$ \Delta S_t = r \cdot S_{t-1}\left(1 - \frac{S_{t-1}}{K}\right) + g(a_t) + \gamma \cdot \text{pulse}_t + \varepsilon_t $$



$$ \Delta S_t = r_{\mathrm{seg}(t)} S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right) + \gamma_{\mathrm{pulse}}\, \mathrm{pulse}_t + \gamma_{\mathrm{step}}\, \mathrm{step}_t + \gamma_{\mathrm{exog}} x_t + \varepsilon_t $$

$$
\Delta S_t =
r_{\mathrm{seg}(t)} S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right)
+ \gamma_{\mathrm{pulse}}\, \mathrm{pulse}_t
+ \gamma_{\mathrm{step}}\, \mathrm{step}_t
+ \gamma_{\mathrm{exog}}\, x_t
+ \gamma_{\mathrm{intercept}}
+ \varepsilon_t
$$


$$ \Delta S_t = \underbrace{r_{\text{seg}(t)} S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right)}_{\text{endogenous logistic growth}} + \underbrace{\gamma_{\text{pulse}} \text{pulse}_t}_{\text{temporary shock}} + \underbrace{\gamma_{\text{step}} \text{step}_t}_{\text{persistent level shift}} + \underbrace{\gamma_{\text{exog}} x_t}_{\text{external regressor effect}} + \underbrace{\varepsilon_t}_{\text{random noise/residual}} $$




St - subscribers at time t

S\_{t-1} - subscribers at time t-1

r - growth rate

K - carrying capacity

γ_exog - ad effectiveness
* This is the key term when deciding how much impact an ad has. If you're trying to estimate the impact of future ads, this term is key.
* x_t represents the amount of advertising input at time 

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

### Sample values:

#### Segment growth rate

| r | Approx. doubling time* | Growth label | What it looks like |
|---|---|---|---|
| 0.03 – 0.07 | 10–25 mo | Slow burn | Steady organic word-of-mouth only |
| 0.10 – 0.20 | 4–7 mo | Healthy growth | Active content + occasional shout-outs |
| 0.25 – 0.40 | 2–3 mo | Fast / breakout | Frequent mentions, algorithmic boosts |
| > 0.5 | < 2 mo | Hyper-growth | Major virality or large paid push |


#### Exogenous γ-parameters

These are multipliers for events or external variables.

| Parameter | Role | Typical Scale (Δ Subscribers per event) | Notes |
|---|---|---|---|
| γₚᵤₗₛₑ | Temporary one-month bump | 100 – 5,000 | Spikes from a shout-out or ad; effect fades next month |
| γₛₜₑₚ | Permanent level shift | 50 – 2,000 | New baseline after paywall change, rebrand, or cross-promotion |
| γₑₓₒg | Slope for numeric regressor xₜ | depends on scaling | e.g., +5 subs per $100 ad spend |
| γ₀ (intercept) | Constant drift term | 0 – 20 subs / month | Baseline steady inflow even with no events |

If these are near zero, it means your observed dynamics are well-explained by the endogenous logistic term alone.


#### Interpreting Overall Regimes

| Pattern | Parameter signature | Typical narrative |
|---|---|---|
| Organic maturity | Small r, no γs | Word-of-mouth plateau |
| Launch buzz | Large early r, γₚᵤₗₛₑ > 0 | One-off viral spike |
| Structural boost | γₛₜₑₚ > 0 | New product line, permanent visibility |
| Ad-driven | γₑₓₒg > 0 tied to spend | Paid campaigns controlling growth |
| Decay phase | Later segment r → 0 | Market saturated, churn ≈ adds |

## Starting Phase 2





