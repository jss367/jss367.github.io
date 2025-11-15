
There are a lot of ways to do project subscriber growth and a lot of design choices. In part, it depends on what levers you control, i.e., what your inputs are. In this post, I'll show how to work with subscriber growth as it comes out of Substack, although it should work in many cases. This will be a somewhat opinionated guide that talks about one way of doing it, even though there are many reasonable options.

## Overall Structure

We'll do this in two phases. Phase 1 will be about determining the best-fit line from historical data. Phase 2 will be about using the results from phase 2 to project future growth.

### Goals for Phase 1

* Normalizes input data

* Find equation
  * finds breakpoints

### Goals for Phase 2

* Be able to answer questions like, "If I spend $1,000 on ads, how much revenue should I expect in return?"

## The Equation

Let's talk about the equation. This equation is the engine behind our model.

$$ \Delta S_t = r_{\mathrm{seg}(t)} S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right) + \gamma_{\mathrm{pulse}}\, \mathrm{pulse}_t + \gamma_{\mathrm{step}}\, \mathrm{step}_t + \gamma_{\mathrm{exog}}\, x_t + \gamma_{\mathrm{intercept}} + \varepsilon_t $$

Let's explain each term.

$$ \Delta S_t = \underbrace{r_{\text{seg}(t)} S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right)}_{\text{endogenous logistic growth}} + \underbrace{\gamma_{\text{pulse}} \text{pulse}_t}_{\text{temporary shock}} + \underbrace{\gamma_{\text{step}} \text{step}_t}_{\text{persistent level shift}} + \underbrace{\gamma_{\text{exog}} x_t}_{\text{external regressor effect}} + \underbrace{\gamma_{\mathrm{intercept}}}_{\text{intercept term}} + \underbrace{\varepsilon_t}_{\text{random noise/residual}} $$

Or, in even more detail:

$$\underbrace{\Delta S_t}_{\text{change in state}} = \underbrace{r_{\mathrm{seg}(t)}}_{\text{growth rate}} \underbrace{S_{t-1}}_{\text{previous state}} \underbrace{\left(1 - \frac{S_{t-1}}{K}\right)}_{\text{carrying capacity}} + \underbrace{\gamma_{\mathrm{pulse}}}_{\text{pulse coefficient}} \underbrace{\mathrm{pulse}_t}_{\text{pulse indicator}} + \underbrace{\gamma_{\mathrm{step}}}_{\text{step coefficient}} \underbrace{\mathrm{step}_t}_{\text{step indicator}} + \underbrace{\gamma_{\mathrm{exog}}}_{\text{exogenous coefficient}} \underbrace{x_t}_{\text{exogenous variable}} + \underbrace{\gamma_{\mathrm{intercept}}}_{\text{intercept term}} + \underbrace{\varepsilon_t}_{\text{error term}}$$

For our simulator, we're not going to have an error term. There will be noise in the data, but adding it to a simulator makes the tool harder to use and doesn't add value, so we'll drop it. So it'll look like this:

$$ \Delta S_t = \underbrace{r_{\text{seg}(t)} S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right)}_{\text{endogenous logistic growth}} + \underbrace{\gamma_{\text{pulse}} \text{pulse}_t}_{\text{temporary shock}} + \underbrace{\gamma_{\text{step}} \text{step}_t}_{\text{persistent level shift}} + \underbrace{\gamma_{\text{exog}} x_t}_{\text{external regressor effect}} + \underbrace{\gamma_{\mathrm{intercept}}}_{\text{intercept term}} $$

$$ \Delta S_t = \underbrace{r_{\text{seg}(t)} S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right)}_{\text{endogenous logistic growth}} + \underbrace{\gamma_{\text{pulse}}\, \text{pulse}_t}_{\text{temporary shock}} + \underbrace{\gamma_{\text{step}}\, \text{step}_t}_{\text{persistent level shift}} + \underbrace{\gamma_{\text{exog}}\, x_t}_{\text{external regressor effect}} + \underbrace{\gamma_{\text{intercept}}}_{\text{baseline organic adds}} $$



St - subscribers at time t

S\_{t-1} - subscribers at time t-1

r - growth rate

K - carrying capacity

γ_exog - ad effectiveness
* This is the key term when deciding how much impact an ad has. If you're trying to estimate the impact of future ads, this term is key.
* x_t represents the amount of advertising input at time 

γpulse - pulse effect

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
| γ_pulse | Temporary one-month bump | 100 – 5,000 | Spikes from a shout-out or ad; effect fades next month |
| γ<sub>pulse</sub> | Temporary one-month bump | 100 – 5,000 | Spikes from a shout-out or ad; effect fades next month |
| v_pulse | Temporary one-month bump | 100 – 5,000 | Spikes from a shout-out or ad; effect fades next month |
| γ_{pulse} | Temporary one-month bump | 100 – 5,000 | Spikes from a shout-out or ad; effect fades next month |
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


# How Ads work

When you upload a spreadsheet, it is used to calculate the lambda and theta values that control the adstock recursion. Adstock is the lingering effect of advertising that continues to influence consumers after the initial ad exposure has ended. The result will be a pandas series sometimes called `exog` that contains an ad effect value for each date. That way, a big ad campaign in January can still be given credit for some of the subscriber ads that are added in February.

There are different ways to find lambda and theta. For example, you could do grid search and see which returns the lowest SSE.

So, you get this pandas series, which is stored in features_df["ad_effect_log"].

If you're testing whether your ads had any effect, you want to start by making sure you get some values in here.

Then, those values can be fit into a function that fits a model to your data.

Here's a table to help understand those values:

| Scenario | λ range | Carryover intuition | θ range | Diminishing-returns intuition | What it usually signals |
| --- | --- | --- | --- | --- | --- |
| “Pulse-only” bursts | 0.0 – 0.1 | Spend decays almost immediately; next month depends only on new spend. | 100 – 250 | Log feature climbs quickly even on modest adstock, so the model will fit strong short-lived bumps if the data warrants it. | Campaigns that hit hard in the launch month (flash sales, single newsletters) and fade fast. |
| Short-lived awareness | 0.2 – 0.4 | A noticeable fraction of last month’s spend lingers, but the signal halves within a couple of months. | 250 – 500 | The log transform still boosts moderate spend, yet allows some headroom for bigger months. | Seasonal pushes or quarterly tests where you expect two to three months of halo effect. |
| Sustained nurture | 0.5 – 0.7 | Carryover rivals fresh spend; it takes several months for the effect to wash out. | 500 – 1000 | The feature grows more gently, implying you need sizable sustained spend to move the needle. | Always-on retargeting or drip campaigns that keep prospects warm for a while. |
| Always-on brand | 0.8 – 0.9 | Most of last month’s value persists; adstock behaves like a smoothed cumulative spend. | 1000 – 2000 | Very slow growth of the log feature, so the model attributes impact only when the spend is both large and persistent. | Mass-awareness or brand advertising where long-term presence matters more than any single pulse. |



`gamma_exog` is a little hard to understand. Because the deltas are in subscribers, gamma_exog is “subscribers gained per unit of the log-transformed ad feature.” A positive value means ads are associated with higher month-to-month growth; a negative value implies the opposite.

So, to figure out how impactful something is going to be, you have to find the log-transformed ad feature.

Here's a guide for this value:

| Magnitude band             | Subscriber impact (ad_effect_log ≈ 1)       | Scenario context                                                        | Interpretation                                                     |
| -------------------------- | ------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Very small (≈0.1 or below) | ≈0.1 subscribers gained per month           | “Ads have no effect” scenario with fitted γ_exog constrained to [−1, 1] | Effectively no detectable ad-driven lift.                          |
| Moderate to strong (≈1–20) | Several extra subscribers per month         | “Ads really valuable” scenario expecting γ_exog between 0.05 and 20     | Meaningful ad contribution without runaway growth.                 |
| Extremely large (≈80–120+) | Dozens to hundreds of subscribers per month | “Spiky spend” benchmark targeting γ_exog between 80 and 120             | Ads dominate growth; only observed in intentionally extreme tests. |




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


# Simulating the future


$$ S_t = S_{t-1} + \gamma_{\text{int}} + r_{\text{last}} S_{t-1}(1 - S_{t-1}/K) + \gamma_{\text{exog}} x_{t-\text{lag}}^{\log} $$


In short, it carries the fitted intercept forward and applies the log-adstock regressor (if present) but omits pulse/step shocks during the forecast horizon.

When you build a forward forecast from a PiecewiseLogisticFit, include the final segment’s intercept 


