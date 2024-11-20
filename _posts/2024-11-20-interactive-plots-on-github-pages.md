---
layout: post
title: "Interactive Plots on Github Pages"
description: "An interactive visualization deployed on Github Pages using Plotly.js"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/auklet.jpg"
tags: [Data Visualization, Github Pages, Interactive Visualizations]
---

This post shows how to create interactive visualizations on Github Pages. This uses Plotly.js.

<div class="controls">
    <div class="control-group">
        <label for="effectSize">Effect Size:</label>
        <input type="range" id="effectSize" min="0" max="2" step="0.1" value="0.5">
        <span id="effectSizeValue">0.5</span>
    </div>
    <div class="control-group">
        <label for="sampleSize">Sample Size:</label>
        <input type="range" id="sampleSize" min="5" max="100" step="5" value="30">
        <span id="sampleSizeValue">30</span>
    </div>
    <div class="control-group">
        <label for="alpha">Alpha:</label>
        <input type="range" id="alpha" min="0.01" max="0.10" step="0.01" value="0.05">
        <span id="alphaValue">0.05</span>
    </div>
</div>
<div id="plot"></div>

<style>
    .controls {
        margin: 20px 0;
        display: flex;
        gap: 20px;
        align-items: center;
        flex-wrap: wrap;
    }
    .control-group {
        display: flex;
        flex-direction: column;
        gap: 5px;
    }
    #plot {
        width: 100%;
        height: 600px;
        margin-bottom: 20px;
    }
    input[type="range"] {
        width: 200px;
    }
</style>

<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.1/plotly.min.js"></script>
<script>
    document.addEventListener('DOMContentLoaded', function() {
        function normalPDF(x, mean, sd) {
            return Math.exp(-0.5 * Math.pow((x - mean) / sd, 2)) / (sd * Math.sqrt(2 * Math.PI));
        }

        function erf(x) {
            const a1 =  0.254829592;
            const a2 = -0.284496736;
            const a3 =  1.421413741;
            const a4 = -1.453152027;
            const a5 =  1.061405429;
            const p  =  0.3275911;

            const sign = (x >= 0) ? 1 : -1;
            x = Math.abs(x);

            const t = 1.0/(1.0 + p*x);
            const y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*Math.exp(-x*x);

            return sign*y;
        }

        function erfInv(x) {
            const a = 0.147;
            const b = 2/(Math.PI * a) + Math.log(1-x*x)/2;
            const sqrt1 = Math.sqrt(b*b - Math.log(1-x*x)/a);
            const sqrt2 = Math.sqrt(sqrt1 - b);
            return sqrt2 * Math.sign(x);
        }

        function createPowerVisualization(effectSize, n, alpha) {
            const se = Math.sqrt(2/n);
            const critValue = -se * Math.sqrt(2) * erfInv(2 * (1 - alpha/2) - 1);
            
            const x = [];
            const nullDist = [];
            const altDist = [];
            
            for (let i = -4*se; i <= 4*se + effectSize; i += se/50) {
                x.push(i);
                nullDist.push(normalPDF(i, 0, se));
                altDist.push(normalPDF(i, effectSize, se));
            }

            const power = 1 - (0.5 * (1 + erf((critValue - effectSize)/(se * Math.sqrt(2)))));

            const data = [
                {
                    x: x,
                    y: nullDist,
                    name: 'Null Distribution',
                    type: 'scatter',
                    mode: 'lines',
                    fill: 'tozeroy',
                    line: {color: 'blue'},
                    fillcolor: 'rgba(0,0,255,0.1)'
                },
                {
                    x: x,
                    y: altDist,
                    name: 'Alternative Distribution',
                    type: 'scatter',
                    mode: 'lines',
                    fill: 'tozeroy',
                    line: {color: 'red'},
                    fillcolor: 'rgba(255,0,0,0.1)'
                },
                {
                    x: [critValue, critValue],
                    y: [0, Math.max(...nullDist, ...altDist)],
                    type: 'scatter',
                    mode: 'lines',
                    line: {dash: 'dash', color: 'gray'},
                    showlegend: false
                },
                {
                    x: [-critValue, -critValue],
                    y: [0, Math.max(...nullDist, ...altDist)],
                    type: 'scatter',
                    mode: 'lines',
                    line: {dash: 'dash', color: 'gray'},
                    showlegend: false
                }
            ];

            const layout = {
                title: `Statistical Power Visualization<br>Power = ${power.toFixed(3)}`,
                xaxis: {title: 'Test Statistic'},
                yaxis: {title: 'Density'},
                hovermode: 'x'
            };

            Plotly.newPlot('plot', data, layout);
        }

        function updatePlot() {
            const effectSize = parseFloat(document.getElementById('effectSize').value);
            const sampleSize = parseInt(document.getElementById('sampleSize').value);
            const alpha = parseFloat(document.getElementById('alpha').value);

            document.getElementById('effectSizeValue').textContent = effectSize.toFixed(1);
            document.getElementById('sampleSizeValue').textContent = sampleSize;
            document.getElementById('alphaValue').textContent = alpha.toFixed(2);

            createPowerVisualization(effectSize, sampleSize, alpha);
        }

        // Set up event listeners
        document.getElementById('effectSize').addEventListener('input', updatePlot);
        document.getElementById('sampleSize').addEventListener('input', updatePlot);
        document.getElementById('alpha').addEventListener('input', updatePlot);

        // Initial plot
        updatePlot();
    });
</script>


