


Here's the equation you need:

![[Pasted image 20250930210506.png]]


Let's do it for my graph:


You might wonder about gat vs gamma pulse. If you buy a bunch of ads, that's gat, right? But if substack features you for free, that's gamma_pulse. mathematically they both just add terms into the state equation, so why not lump them together? The distinction is less about the arithmetic and more about **modeling control vs. exogenous shocks**.



OK, so if you look at a graph, you need to estimate the following:
growth rate
carrying capacity
shocks (internal and external)



There will be structural changes, so you'll want a piecewise function.


![[Pasted image 20251006213301.png]]



```
\[
\Delta S_t =
\underbrace{r_{\mathrm{seg}(t)} S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right)}_{\text{endogenous/logistic growth (segment r)}}
+ \underbrace{\gamma_{\mathrm{pulse}}\, \mathrm{pulse}_t}_{\text{one-month shock}}
+ \underbrace{\gamma_{\mathrm{step}}\, \mathrm{step}_t}_{\text{persistent additive drift}}
+ \gamma_{\mathrm{exog}} x_t
\]
```


## Why keep both “effect” and “component”?

Think of them as orthogonal:

- **effect** = _persistence of the shock_ (what goes in):
    
    - **Transient** → maps to **pulse** term
        
    - **Persistent** → maps to **step** term
        
    - **No effect** → ignore
        
- **component** = _what kind of change we think it is_:
    
    - **rate** → change in rrr → **create segment boundary**
        
    - **level** → **step** (persistent additive)
        
    - **pulse** → **pulse** (one-off)
        
    - **mixed** → both (e.g., big one-off plus ongoing slope change)



Confusion here:

Does transient mean it comes back down? But that's not going to happen (I don't think). If it's a one time thing, then it's a step change.

- **Transient** → maps to **pulse** term
    
- **Persistent** → maps to **step** term




- **Transient (one-month shock in ΔS)**  
    You add a spike to _one_ month’s net adds (γ_pulse · pulse_t).
    
    - ΔS: big for that month, back to baseline next month.
        
    - **S**: jumps up and **stays higher** (it does _not_ “come back down”) unless churn/negatives remove it later.
        
    - **r** (the logistic coefficient) does **not** change.
        
- **Persistent (ongoing uplift in ΔS)**  
    You add a constant offset every month from that date (γ_step · step_t).
    
    - ΔS: permanently higher by ~γ_step each month.
        
    - **S**: slope is steeper thereafter (additive drift).
        
    - **r** still does **not** change.
        
- **Segment change (r changes)**  
    The **multiplicative growth law** changes: the coefficient on xt=St−1(1−St−1/K)x_t=S_{t-1}(1-S_{t-1}/K)xt​=St−1​(1−St−1​/K) is different after the date.
    
    - ΔS: scales differently _with S_.
        
    - **S**: curvature/trajectory changes.
        
    - This is **not** encoded by “Transient”/“Persistent”; it’s a **breakpoint** that starts a new r segment.


Wait, so persistent change is a way of saying that you're growing at a faster rate, but r hasn't changed? This is confusing.


![[Pasted image 20251006214539.png]]


- A Friday tweet drives **+1,000** subs that month and then disappears.  
    Model: **Transient pulse** at that month. **No segment.**  
    S jumps and stays; ΔS returns to baseline next month.
    
- You turn on a **new steady referral source** adding ~+80 net subs every month.  
    Model: **Persistent step** from that date. **No segment.**  
    ΔS baseline is 80 higher forever; r unchanged.
    
- Your audience doubles and **virality per subscriber improves** (e.g., stronger word-of-mouth because of a network effect).  
    Model: likely **mixed** — a **pulse** (the import/merger month), **maybe a step**, **and a new segment r** because the _proportional_ response to S changed.



### Ad Spend

Think of **ad spend** as an **additive driver of ΔS** (monthly net adds), not as a change to the growth law r.


## Default (recommended): exogenous ads path

- **What it is:** Upload a `date_spend` CSV → app builds `adstock` (λ) and `ad_effect_log = log(1 + adstock/θ)` → fit with **γ_exog · x_t**.
    
- **Effect/component labels:** In Events, either don’t add ad events at all, or add them as **“No effect”** (for visual markers only). The _actual impact_ flows through **x_t**.
    
- **Why:** This captures campaigns, carryover, and scale with spend; when spend stops, the effect decays with λ. No double counting.
    

## If you don’t have a spend CSV (events-only fallback)

- Enter one **“Ad spend” event per month of the campaign** with:
    
    - **Effect:** **Transient** (one-month shock in ΔS),
        
    - **cost:** set ≈ that month’s spend (your code already weights ad pulses by `cost`).
        
- Set **λ** in features so adstock gives you realistic carryover.
    
- **Do not** use **Persistent** for ads: that would add a permanent offset to ΔS even after spend stops.





We rarely change r, although it's possible.



## Debug


![[Pasted image 20251015083352.png]]


