


There are a lot of ways to do this and it depends on what your inputs are. In this post I'll show how to work with subscriber growth as it comes out of Substack, although it should work in many cases.

## Goals from Phase 1:

* Normalizes input data

* Find equation
  * finds breakpoints


## The Equation

Let's talk about the equation

$$ \Delta S_t = r \cdot S_{t-1}\left(1 - \frac{S_{t-1}}{K}\right) + g(a_t) + \gamma \cdot \text{pulse}_t + \varepsilon_t $$



$$ \Delta S_t = r_{\mathrm{seg}(t)} S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right) + \gamma_{\mathrm{pulse}}\, \mathrm{pulse}_t + \gamma_{\mathrm{step}}\, \mathrm{step}_t + \gamma_{\mathrm{exog}} x_t + \varepsilon_t $$

$$
\Delta S_t = r_{\mathrm{seg}(t)} S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right) + \gamma_{\mathrm{pulse}}\, \mathrm{pulse}_t + \gamma_{\mathrm{step}}\, \mathrm{step}_t + \gamma_{\mathrm{exog}}\, x_t + \gamma_{\mathrm{intercept}} + \varepsilon_t $$


$$ \Delta S_t = \underbrace{r_{\text{seg}(t)} S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right)}_{\text{endogenous logistic growth}} + \underbrace{\gamma_{\text{pulse}} \text{pulse}_t}_{\text{temporary shock}} + \underbrace{\gamma_{\text{step}} \text{step}_t}_{\text{persistent level shift}} + \underbrace{\gamma_{\text{exog}} x_t}_{\text{external regressor effect}} + \underbrace{\varepsilon_t}_{\text{random noise/residual}} $$

$$ \Delta S_t =
\underbrace{r_{\text{seg}(t)} S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right)}_{\text{endogenous logistic growth}} + \underbrace{\gamma_{\text{pulse}}\, \text{pulse}_t}_{\text{temporary shock}} + \underbrace{\gamma_{\text{step}}\, \text{step}_t}_{\text{persistent level shift}} + \underbrace{\gamma_{\text{exog}}\, x_t}_{\text{external regressor effect}} + \underbrace{\gamma_{\text{intercept}}}_{\text{baseline organic adds}} + \underbrace{\varepsilon_t}_{\text{random noise/residual}} $$



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

## Ending Phase 1

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
| γₑₓₒg | Slope for numeric regressor xₜ. THIS IS WHERE THE VALUE OF ADS COMES IN | depends on scaling | e.g., +5 subs per $100 ad spend |
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




# What to Expect

| Growth driver / event | Parameters most directly affected | Why it maps to those levers |
| --- | --- | --- |
| One-off shout-out, podcast mention, newsletter swap | Pulse term γ via `pulse_t` | These are short-lived surges in attention, modeled as transient pulses that scale with γ before decaying. |
| Cross-promotion series with another creator (multi-email swap) | Pulse term γ; potentially baseline rate r if the partnership is ongoing | Each coordinated drop yields temporary spikes (γ·pulse_t); if the collaboration turns into a sustained referral channel, it can permanently lift the organic growth rate r. |
| Single viral post or press hit | Pulse term γ | Viral moments cause temporary subscriber jumps captured by the pulse component rather than a structural shift in the logistic core. |
| One-time ad campaign or sponsored placement | Ad response g(a_t): β and θ; adstock a_t via spend input; short-term effect through λ-controlled carryover | A discrete buy increases ad spend x_t, which flows into adstock a_t and the diminishing-returns transform g(a_t); the decay of that spend is governed by λ, so the immediate impact lives in these parameters rather than the baseline logistic term. |
| Sustained ad campaign / always-on acquisition | Ad response g(a_t): β, θ; adstock memory λ | A long-running program keeps a_t elevated, so the response curve parameters (β, θ) and the carryover factor λ determine how strongly sustained spend translates into steady subscriber additions. |
| Writing better content / improved value proposition | Baseline growth rate r; potential carrying capacity K | Higher quality increases organic referrals and retention, boosting r; a meaningfully better product can raise the eventual ceiling K by expanding the reachable audience before saturation. |
| Publishing more frequently / consistent cadence | Baseline growth rate r | More touchpoints typically compound organic sharing and conversions, raising the intrinsic growth rate r in the logistic term. |
| Change in paywall strategy (e.g., moving more posts free or more paid) | Carrying capacity K; possibly baseline r | Adjusting the paywall changes how many people are willing to subscribe at all (K) and how readily they convert from exposure (r). A tighter paywall can shrink the reachable base; a looser one can expand it. |
| Rebrand / repositioning of the publication | Baseline growth rate r; carrying capacity K | A successful rebrand can refresh word-of-mouth (raising r) and open a bigger addressable audience (raising K); conversely, a misfire could reduce both. |
| Strategic partnership or integration (e.g., bundling with another creator long-term) | Baseline growth rate r; possibly g(a_t) if partner spend is tracked as adstock | Persistent referral inflows effectively shift the organic trajectory captured by r; if you model the partner’s promotions as ongoing “spend,” they also live in g(a_t)’s parameters. |
| Platform-level feature (Substack homepage feature, curated list inclusion) | Pulse term γ for limited-time boosts; baseline r if the feature becomes permanent | Temporary placement acts like a strong pulse; if the platform keeps you highlighted over months, it can permanently change your discovery rate, nudging r upward. |
| Significant product improvement outside publishing (e.g., community launch, better onboarding) | Baseline growth rate r; carrying capacity K | Improved onboarding raises the share of visitors who subscribe (higher r) and may expand how many people stick around before saturation (higher K). |

