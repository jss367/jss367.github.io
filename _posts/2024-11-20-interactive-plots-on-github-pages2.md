---
layout: post
title: "Understanding Statistical Power: An Interactive Guide"
description: "An interactive visualization of statistical power concepts using Plotly.js"
feature-img: "assets/img/hummingbird.jpg"
thumbnail: "assets/img/hummingbird.jpg"
tags: [Statistics, Data Science, Python, Interactive]
---

# Understanding Statistical Power: An Interactive Guide

Statistical power is a fundamental concept in research design and hypothesis testing. This interactive guide will help you build intuition about how different factors affect statistical power.

<div id="power-visualization">
    <style>
        .controls {
            margin: 20px 0;
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        #plot {
            width: 100%;
            height: 600px;
        }
    </style>
    
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

    <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.1/plotly.min.js"></script>
    <script>
        // [Previous JavaScript code goes here]
        function normalPDF(x, mean, sd) {
            return Math.exp(-0.5 * Math.pow((x - mean) / sd, 2)) / (sd * Math.sqrt(2 * Math.PI));
        }

        // [Rest of the JavaScript code from the previous HTML file]
        // [Make sure to include all the functions: createPowerVisualization, erf, erfInv, updatePlot]
    </script>
</div>

## What is Statistical Power?

Statistical power is the probability of detecting a true effect when one exists. In other words, it's the likelihood that your study will find a statistically significant result when there really is a difference or effect to be found.