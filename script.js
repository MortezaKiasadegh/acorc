// Constants
const e = 1.6e-19;

// Cunningham slip correction factor
function Cc(d, P, T) {
    const la = 67.3e-9; // mfp at 101325 Pa and 296.15 K
    const lap = la * Math.pow(T / 296.15, 2) * (101325 / P) * ((110.4 + 296.15) / (T + 110.4));
    const alpha = 1.165 * 2;
    const beta = 0.483 * 2;
    const gamma = 0.997 / 2;
    return 1 + lap / d * (alpha + beta * Math.exp(-gamma * d / lap));
}

// Helper function for numerical optimization
function leastSquares(func, guess, bounds, args) {
    const maxIterations = 500;
    const tolerance = 1e-8;
    let x = guess;
    let step = Math.abs(guess) * 0.1;
    let bestX = guess;
    let bestError = Infinity;
    
    // Try different initial guesses
    const initialGuesses = [guess, guess * 0.1, guess * 10, guess * 0.01, guess * 100];
    
    for (const initialGuess of initialGuesses) {
        x = initialGuess;
        step = Math.abs(initialGuess) * 0.1;
        
        for (let i = 0; i < maxIterations; i++) {
            try {
                const fx = func(x, ...args);
                
                if (!isFinite(fx)) {
                    console.warn(`Non-finite value at x=${x.toExponential(6)}`);
                    break;
                }
                
                if (Math.abs(fx) < bestError) {
                    bestError = Math.abs(fx);
                    bestX = x;
                }
                
                if (Math.abs(fx) < tolerance) {
                    console.log(`Converged at x=${x.toExponential(6)} with error=${fx.toExponential(6)}`);
                    return { x, error: fx };  // ✅ Matches expected structure
                }
                
                // Simple gradient descent with adaptive step size
                const fxPlus = func(x + step, ...args);
                const fxMinus = func(x - step, ...args);
                
                if (!isFinite(fxPlus) || !isFinite(fxMinus)) {
                    step *= 0.5;
                    continue;
                }
                
                const derivative = (fxPlus - fxMinus) / (2 * step);
                
                if (Math.abs(derivative) < 1e-10) {
                    step *= 2;
                    continue;
                }
                
                const newX = x - fx / derivative;
                
                if (!isFinite(newX)) {
                    step *= 0.5;
                    continue;
                }
                
                // Ensure we stay within bounds
                x = Math.max(bounds[0], Math.min(bounds[1], newX));
                
                // Adjust step size based on progress
                if (Math.abs(newX - x) < step * 0.1) {
                    step *= 0.5;
                } else if (Math.abs(newX - x) > step * 0.9) {
                    step *= 2;
                }
                
            } catch (error) {
                console.warn(`Optimization step ${i} failed:`, error);
                step *= 0.5;
            }
        }
    }
    
    console.log(`Best solution found: x=${bestX.toExponential(6)} with error=${bestError.toExponential(6)}`);
    return { x: bestX, error: bestError };
}

function levenbergMarquardt1D(func, guess, args = [], bounds = [1e-9, 5e-6], {
    lambdaInit = 1e-3,
    tol = 1e-12,
    maxIter = 1000,
    h = 1e-8
} = {}) {
    let x = guess;
    let lambda = lambdaInit;

    for (let i = 0; i < maxIter; i++) {
        const fx = func(x, ...args);
        const fxPlus = func(x + h, ...args);
        const dfdx = (fxPlus - fx) / h;

        if (!isFinite(fx) || !isFinite(dfdx) || Math.abs(dfdx) < 1e-20) break;

        let step = fx / (dfdx + lambda);
        if (!isFinite(step)) break;

        let x_new = x - step;
        x_new = Math.min(bounds[1], Math.max(bounds[0], x_new));  // clamp to bounds
        const fx_new = func(x_new, ...args);

        if (!isFinite(fx_new)) {
            lambda *= 10;
            continue;
        }

        // Update if better
        if (Math.abs(fx_new) < Math.abs(fx)) {
            x = x_new;
            lambda *= 0.7; // Reduce damping if successful
        } else {
            lambda *= 2.5; // Increase damping if not
        }

        if (Math.abs(fx_new) < tol) {
            return x_new;
        }
    }

    console.warn("LM optimizer exited without full convergence.");
    return x;
}



function calculateCPMARange() {
    try {
        // Get input values
        const Q_a = parseFloat(document.getElementById('Q_a').value) / 60000; // Convert to m³/s
        const R_m = parseFloat(document.getElementById('R_m').value);
        const r_1 = parseFloat(document.getElementById('r_1').value);
        const r_2 = parseFloat(document.getElementById('r_2').value);
        const L = parseFloat(document.getElementById('L').value);
        const V_min = parseFloat(document.getElementById('V_min').value);
        const V_max = parseFloat(document.getElementById('V_max').value);
        const w_lb = 2 * Math.PI / 60 * parseFloat(document.getElementById('w_lb').value);
        const w_ub = 2 * Math.PI / 60 * parseFloat(document.getElementById('w_ub').value);
        const rho100 = parseFloat(document.getElementById('rho100').value);
        const Dm = parseFloat(document.getElementById('Dm').value);
        const T = parseFloat(document.getElementById('T').value);
        const P = parseFloat(document.getElementById('P').value);

        // Validate input parameters
        if (isNaN(Q_a) || isNaN(R_m) || isNaN(r_1) || isNaN(r_2) || isNaN(L) || 
            isNaN(V_min) || isNaN(V_max) || isNaN(w_lb) || isNaN(w_ub) || 
            isNaN(rho100) || isNaN(Dm) || isNaN(T) || isNaN(P)) {
            throw new Error('Invalid input parameters');
        }

        console.log('Input parameters:', {
            Q_a: Q_a.toExponential(6),
            R_m: R_m.toExponential(6),
            r_1: r_1.toExponential(6),
            r_2: r_2.toExponential(6),
            L: L.toExponential(6),
            V_min: V_min.toExponential(6),
            V_max: V_max.toExponential(6),
            w_lb: w_lb.toExponential(6),
            w_ub: w_ub.toExponential(6),
            rho100: rho100.toExponential(6),
            Dm: Dm.toExponential(6),
            T: T.toExponential(6),
            P: P.toExponential(6)
        });

        // Calculate gas properties
        const mu = 1.81809e-5 * Math.pow(T / 293.15, 1.5) * (293.15 + 110.4) / (T + 110.4);

        // Classifier parameters
        const r_c = (r_1 + r_2) / 2;
        const log_r_ratio = Math.log(r_2 / r_1);

        // Mass-mobility parameters
        const k = Math.PI / 6 * rho100 * Math.pow(100 * 1e-9, 3 - Dm);
        const m_low = 1e-22;
        const m_up = 1e-14;
        const m_star = Array.from({length: 1000}, (_, i) => 
            Math.pow(10, Math.log10(m_low) + (Math.log10(m_up) - Math.log10(m_low)) * i / 999)
        );
        const d_m_spa = m_star.map(m => Math.pow(m/k, 1/Dm));

        console.log('Calculation parameters:', {
            mu: mu.toExponential(6),
            r_c: r_c.toExponential(6),
            log_r_ratio: log_r_ratio.toExponential(6),
            k: k.toExponential(6),
            m_star_range: [m_star[0].toExponential(6), m_star[m_star.length-1].toExponential(6)],
            d_m_spa_range: [d_m_spa[0].toExponential(6), d_m_spa[d_m_spa.length-1].toExponential(6)]
        });

        // Precompute factors
        const factor1 = e * V_min / (k * r_c**2 * log_r_ratio);
        const factor2 = e * V_max / (k * r_c**2 * log_r_ratio);
        const factor3 = 3 * mu * Q_a / (2 * k * r_c**2 * L);

        console.log('Factors:', {
            factor1: factor1.toExponential(6),
            factor2: factor2.toExponential(6),
            factor3: factor3.toExponential(6)
        });

        // Calculate R_m sweep
        const R_m_1 = [];
        const R_m_2 = [];
        const valid_d_m = [];


        function residual2(d_m_max, d_m_val, factor_v) {
            try {
                const factor = factor_v === 'min' ? factor1 : factor2;
                let w_guess = Math.pow(factor / Math.pow(d_m_val, Dm), 0.5);
                const w = Math.max(w_lb, Math.min(w_ub, w_guess));
                
                const Cc_val = Cc(d_m_max, P, T);
                const res = Math.pow(d_m_max, Dm) - Math.pow(d_m_val, Dm) - 
                           (factor3 / Math.pow(w, 2)) * (d_m_max / Cc_val);
                
                return res / Math.abs(Math.pow(d_m_max, Dm));
            } catch (error) {
                console.warn('Error in residual2 function:', error);
                return 1e10; // Return a large value to indicate error
            }
        }

        function optimizeDiameter2(d_m_val, factor_v, guess) {
            const { x, error } = leastSquares(
                residual2,
                guess,
                [1e-9, 5e-5],
                [d_m_val, factor_v]
            );
        
            if (typeof x !== 'number' || !isFinite(x) || isNaN(x)) {
                console.warn("optimizeDiameter2: invalid x");
                return NaN;
            }
        
            if (!isFinite(error) || Math.abs(error) > 1e-4) {
                console.warn(`optimizeDiameter2: Ignored result x=${x.toExponential(6)}, error=${error.toExponential(6)}`);
                return NaN;
            }
        
            return x;
        }

        // First, calculate the R_m sweep
        for (let i = 0; i < d_m_spa.length; i++) {
            const d_m_val = d_m_spa[i];
            try {
                const d_m_max_1 = optimizeDiameter2(d_m_val, 'min', d_m_val);
                const d_m_max_2 = optimizeDiameter2(d_m_val, 'max', d_m_val);

                const R_m_val_1 = Math.pow(d_m_val, Dm) / (Math.pow(d_m_max_1, Dm) - Math.pow(d_m_val, Dm));
                const R_m_val_2 = Math.pow(d_m_val, Dm) / (Math.pow(d_m_max_2, Dm) - Math.pow(d_m_val, Dm));

                if (Math.abs(R_m_val_1 - R_m_val_2) > 1e-4 && 
                    !isNaN(R_m_val_1) && !isNaN(R_m_val_2) &&
                    isFinite(R_m_val_1) && isFinite(R_m_val_2) &&
                    R_m_val_1 > 0 && R_m_val_2 > 0) {
                    valid_d_m.push(d_m_val * 1e9);
                    R_m_1.push(R_m_val_1);
                    R_m_2.push(R_m_val_2);
                }
            } catch (error) {
                console.warn(`Error calculating point ${i}:`, error);
            }
        }

        console.log('Valid data points:', valid_d_m.length);
        console.log('Sample data:', {
            valid_d_m: valid_d_m.slice(0, 5).map(x => x.toExponential(6)),
            R_m_1: R_m_1.slice(0, 5).map(x => x.toExponential(6)),
            R_m_2: R_m_2.slice(0, 5).map(x => x.toExponential(6))
        });

        if (valid_d_m.length === 0) {
            throw new Error('No valid data points were calculated. Please check the input parameters.');
        }

        const d_min_valid = valid_d_m[0] * 1e-9;
        const d_max_valid = valid_d_m[valid_d_m.length - 1] * 1e-9;
        
        function residualDirect(d, R_m_val, factor_v) {
            const d_val = d;
            const d_m_max = Math.pow((R_m_val + 1) / R_m_val, 1 / Dm) * d_val;
            const factor = factor_v === 'min' ? factor1 : factor2;
        
            let w_guess = Math.sqrt(factor / Math.pow(d_val, Dm));
            let w = Math.min(w_ub, Math.max(w_lb, w_guess)); // Clamp to bounds
        
            const Cc_val = Cc(d_m_max, P, T);
            const res = Math.pow(d_val, Dm) - Math.pow(d_m_max, Dm) +
                        (factor3 / (w * w)) * (d_m_max / Cc_val);
        
            return res / Math.abs(Math.pow(d_val, Dm));
        }
        
        function optimizeDiameterDirect(R_m_val, factor_v, guess, bounds = [d_min_valid, d_max_valid]) {
            const { x, error } = leastSquares(residualDirect, guess, bounds, [R_m_val, factor_v]);
        
            if (typeof x !== 'number' || isNaN(x)) {
                console.warn("Invalid result object or x is not a number");
                return NaN;
            }
        
            if (!isFinite(error) || Math.abs(error) > 1e-8) {
                console.warn(`Ignored result: x=${x.toExponential(6)}, error=${error.toExponential(6)} > 1e-8`);
                return NaN;
            }
            console.log("Returning from optimizeDiameterDirect:", x);
            return x;
        }

        d_i_1 = optimizeDiameterDirect(R_m, 'min', 1.5e-8)
        d_o_1 = optimizeDiameterDirect(R_m, 'min', 1.5e-6)
        d_i_2 = optimizeDiameterDirect(R_m, 'max', 1.5e-8)
        d_o_2 = optimizeDiameterDirect(R_m, 'max', 1.5e-6)


        if (isNaN(d_i_1)) d_i_1 = d_i_2;
        if (isNaN(d_i_2)) d_i_2 = d_i_1;
        if (isNaN(d_o_1)) d_o_1 = d_o_2;
        if (isNaN(d_o_2)) d_o_2 = d_o_1;

        // Final safety net
        if (isNaN(d_i_1) || isNaN(d_i_2)) {
            console.error("Both d_i roots failed — cannot continue");
            alert("Error: Could not find valid d_i. Check your input or try different parameters.");
            return;
        }
        if (isNaN(d_o_1) || isNaN(d_o_2)) {
            console.error("Both d_o roots failed — cannot continue");
            alert("Error: Could not find valid d_o. Check your input or try different parameters.");
            return;
        }
        // Handle NaNs by falling back to the other result
        function safeMax(a, b) {
            if (isNaN(a)) return b;
            if (isNaN(b)) return a;
            return Math.max(a, b);
        }
        
        function safeMin(a, b) {
            if (isNaN(a)) return b;
            if (isNaN(b)) return a;
            return Math.min(a, b);
        }

        d_i_i = safeMax(d_i_1, d_i_2);
        d_o_o = safeMin(d_o_1, d_o_2);

        d_i = Math.min(d_i_i, d_o_o);
        d_o = Math.max(d_i_i, d_o_o);

        const valid_m = valid_d_m.map(d => k * Math.pow(d * 1e-9, Dm));
        const valid_m_i = k * Math.pow(d_i, Dm);
        const valid_m_o = k * Math.pow(d_o, Dm);

        console.log({
            d_i_1: d_i_1?.toExponential?.(6),
            d_i_2: d_i_2?.toExponential?.(6),
            d_o_1: d_o_1?.toExponential?.(6),
            d_o_2: d_o_2?.toExponential?.(6),
            d_i: d_i?.toExponential?.(6),
            d_o: d_o?.toExponential?.(6)
        });

        console.log({
            d_i_1, d_i_2, d_o_1, d_o_2, d_i, d_o
        });
        
        
        d_i = d_i.toExponential(6);
        d_o = d_o.toExponential(6);

        
                     
        const yAxisMode = document.getElementById('yaxis-mode').value;
        let x_vals, x_label, d_i_label, d_o_label;
        if (yAxisMode === "mass") {
            x_vals = valid_m.map(d => d * 1e18); 
            d_i_label = (valid_m_i * 1e18).toFixed(6); // fg
            d_o_label = (valid_m_o * 1e18).toFixed(6); // fg
            x_label = 'Mass [fg]';
        } else {
            x_vals = valid_d_m; // m → nm
            d_i_label = (d_i * 1e9).toFixed(3); // nm
            d_o_label = (d_o * 1e9).toFixed(3); // nm
            x_label = 'Mobility diameter, Dₘ [nm]';
        }

        document.getElementById('diameter-results_CPMA').innerHTML = `
            <strong>Selected Boundaries:</strong><br>
            ${yAxisMode === 'mass' 
                ? `m_i_CPMA = ${d_i_label} fg<br>m_o_CPMA = ${d_o_label} fg`
                : `d_i_CPMA = ${d_i_label} nm<br>d_o_CPMA = ${d_o_label} nm`}
        `;

        // Plot the results
        const trace1 = {
            x: x_vals,
            y: R_m_1,
            type: 'scatter',
            mode: 'lines',
            showlegend: false,
            line: {color: 'red'}
        };

        const trace2 = {
            x: x_vals,
            y: R_m_2,
            type: 'scatter',
            mode: 'lines',
            showlegend: false,
            line: {color: 'red'}
        };

        const trace3 = {
            x: [d_i_label, d_o_label],
            y: [R_m, R_m],
            type: 'scatter',
            mode: 'lines',
            name: 'Input Rm boundary',
            showlegend: true,
            line: {color: 'green'}
        };

        const layout = {
            title: 'CPMA Operational Range',
            xaxis: {
                title: x_label,
                title: {
                    text: x_label,
                    font: { size: 18 },
                    standoff: 15
                },
                type: 'log'
            },
            yaxis: {
                title: {
                    text: 'Rₘ',
                    font: { size: 18 },
                    standoff: 15
                },
                type: 'log'
            },
            grid: {
                rows: 1,
                columns: 1,
                pattern: 'independent'
            }
        };

        // Check if Plotly is loaded
        if (typeof Plotly === 'undefined') {
            throw new Error('Plotly library is not loaded');
        }

        // Clear previous plot
        const plotDiv = document.getElementById('plot_CPMA');
        plotDiv.innerHTML = '';

        // Create new plot
        Plotly.newPlot('plot_CPMA', [trace1, trace2, trace3], layout)
            .then(() => console.log('Plot created successfully'))
            .catch(err => console.error('Error creating plot:', err));

    } catch (error) {
        console.error('Error in calculateCPMARange:', error);
        alert('An error occurred while calculating the CPMA range. Please check the console for details.');
    }
}

function calculateDMARange() {
    try {
        // Get input values
        const Q_a = parseFloat(document.getElementById('DMA_Q_a').value) / 60000; // Convert to m³/s
        const Q_sh = parseFloat(document.getElementById('DMA_Q_sh').value) / 60000; // Convert to m³/s
        const Q_sh_lb = parseFloat(document.getElementById('DMA_Q_sh_lb').value) / 60000; // Convert to m³/s
        const Q_sh_ub = parseFloat(document.getElementById('DMA_Q_sh_ub').value) / 60000; // Convert to m³/s
        const r_1 = parseFloat(document.getElementById('DMA_r_1').value);
        const r_2 = parseFloat(document.getElementById('DMA_r_2').value);
        const L = parseFloat(document.getElementById('DMA_L').value);
        const V_min = parseFloat(document.getElementById('DMA_V_min').value);
        const V_max = parseFloat(document.getElementById('DMA_V_max').value);
        const T = parseFloat(document.getElementById('DMA_T').value);
        const P = parseFloat(document.getElementById('DMA_P').value);

        // Validate input parameters
        if (isNaN(Q_a) || isNaN(Q_sh) || isNaN(Q_sh_lb) || isNaN(Q_sh_ub)  || isNaN(r_1) || isNaN(r_2) || isNaN(L) || 
            isNaN(V_min) || isNaN(V_max) || isNaN(T) || isNaN(P)) {
            throw new Error('Invalid input parameters');
        }

        console.log('Input parameters:', {
            Q_a: Q_a.toExponential(6),
            Q_sh: Q_sh.toExponential(6),
            Q_sh_lb: Q_sh_lb.toExponential(6),
            Q_sh_ub: Q_sh_ub.toExponential(6),
            r_1: r_1.toExponential(6),
            r_2: r_2.toExponential(6),
            L: L.toExponential(6),
            V_min: V_min.toExponential(6),
            V_max: V_max.toExponential(6),
            T: T.toExponential(6),
            P: P.toExponential(6)
        });

        // Calculate gas properties
        const mu = 1.81809e-5 * Math.pow(T / 293.15, 1.5) * (293.15 + 110.4) / (T + 110.4);

        // Classifier parameters
        const log_r_ratio = Math.log(r_2 / r_1);
        const Q_sh_spa = Array.from({length: 1000}, (_, i) => 
            Math.pow(10, Math.log10(Q_sh_lb) + (Math.log10(Q_sh_ub) - Math.log10(Q_sh_lb)) * i / 999)
        );
        const R_B = Q_sh_spa.map(q => q / Q_a);
        const R_B_lb = Q_sh_lb / Q_a;
        const R_B_ub = Q_sh_ub / Q_a;
        const R_B_i = Q_sh / Q_a;

        console.log('Calculation parameters:', {
            mu: mu.toExponential(6),
            log_r_ratio: log_r_ratio.toExponential(6),
            Q_sh_spa_range: [Q_sh_spa[0].toExponential(6), Q_sh_spa[Q_sh_spa.length-1].toExponential(6)],
            R_B: [R_B[0].toExponential(6), R_B[Q_sh_spa.length-1].toExponential(6)],
            R_B_lb: R_B_lb.toExponential(6),
            R_B_ub: R_B_ub.toExponential(6),
            R_B_i: R_B_i.toExponential(6)
        });

        // Precompute factors
        const factor1 = 2 * e * V_min * L / (3 * mu * log_r_ratio);
        const factor2 = 2 * e * V_max * L / (3 * mu * log_r_ratio);

        console.log('Factors:', {
            factor1: factor1.toExponential(6),
            factor2: factor2.toExponential(6)
        });


        // Root-finder (simple Newton-style)
        function findRoot(func, guess, args = [], bounds = [5e-10, 5e-6], {
            tol = 1e-12,
            maxIter = 500,
            h = 1e-8
        } = {}) {
            let x = guess;
            for (let i = 0; i < maxIter; i++) {
                const fx = func(x, ...args);
                const fxh = func(x + h, ...args);
                const dfdx = (fxh - fx) / h;

                if (Math.abs(dfdx) < 1e-20 || !isFinite(dfdx)) break;

                const step = fx / dfdx;
                x = x - step;
                if (x < bounds[0] || x > bounds[1]) return NaN;

                if (Math.abs(fx) < tol) return x;
            }
            return NaN;
        }

        function f(d, Q_sh_val, factor) {
            return d - (factor / Q_sh_val) * Cc(d, P, T);
        }
    
        const d_min_DMA = Q_sh_spa.map((Q) => findRoot(f, 5e-10, [Q, factor1]));
        const d_max_DMA = Q_sh_spa.map((Q) => findRoot(f, 1e-6, [Q, factor2]));
    
        const d_i = findRoot(f, 5e-10, [Q_sh, factor1]);
        const d_o = findRoot(f, 1e-6, [Q_sh, factor2]);

        console.log({
            d_i: d_i?.toExponential?.(6),
            d_o: d_o?.toExponential?.(6)
        });



        document.getElementById('diameter-results_DMA').innerHTML = `
            <strong>Selected Boundaries:</strong><br>
            d_i_DMA = ${(d_i * 1e9).toFixed(2)} nm<br>
            d_o_DMA = ${(d_o * 1e9).toFixed(2)} nm
        `;
        

        // Plot the results
        const trace1 = {
            x: [d_min_DMA[0] * 1e9, d_max_DMA[0] * 1e9],
            y: [R_B_lb, R_B_lb],
            type: 'scatter',
            mode: 'lines',
            showlegend: false,
            line: {color: 'red'}
        };

        const trace2 = {
            x: [d_min_DMA[Q_sh_spa.length-1] * 1e9, d_max_DMA[Q_sh_spa.length-1] * 1e9],
            y: [R_B_ub, R_B_ub],
            type: 'scatter',
            mode: 'lines',
            showlegend: false,
            line: {color: 'red'}
        };

        const trace3 = {
            x: [d_i * 1e9, d_o * 1e9],
            y: [R_B_i, R_B_i],
            type: 'scatter',
            mode: 'lines',
            name: 'Input Q_sh boundry',
            showlegend: true,
            line: {color: 'green'}
        };

        const trace4 = {
            x: d_min_DMA.map(d => d * 1e9),
            y: R_B,
            type: 'scatter',
            mode: 'lines',
            showlegend: false,
            line: {color: 'red'}
        };

        const trace5 = {
            x: d_max_DMA.map(d => d * 1e9),
            y: R_B,
            type: 'scatter',
            mode: 'lines',
            showlegend: false,
            line: {color: 'red'}
        };

        const layout = {
            title: 'DMA Operational Range',
            xaxis: {
                title: {
                    text: 'Mobility diameter, Dₘ [nm]',
                    font: { size: 18 },
                    standoff: 15
                },
                type: 'log'
            },
            yaxis: {
                title: {
                    text: 'Rₘ',
                    font: { size: 18 },
                    standoff: 15
                },
                type: 'log',
                tickfont: {
                    family: 'Arial, sans-serif'
                }
            },
            grid: {
                rows: 1,
                columns: 1,
                pattern: 'independent'
            }
        };

        // Check if Plotly is loaded
        if (typeof Plotly === 'undefined') {
            throw new Error('Plotly library is not loaded');
        }

        // Clear previous plot
        const plotDiv = document.getElementById('plot_DMA');
        plotDiv.innerHTML = '';

        // Create new plot
        Plotly.newPlot('plot_DMA', [trace1, trace2, trace3, trace4, trace5], layout)
            .then(() => console.log('Plot created successfully'))
            .catch(err => console.error('Error creating plot:', err));

    } catch (error) {
        console.error('Error in calculateCPMARange:', error);
        alert('An error occurred while calculating the CPMA range. Please check the console for details.');
    }

}

function calculateAACRange() {
    try {
        // Get input values
        const Q_a = parseFloat(document.getElementById("AAC_Q_a").value) / 60000;
        const Q_sh = parseFloat(document.getElementById("AAC_Q_sh").value) / 60000;
        const Q_sh_lb = parseFloat(document.getElementById("AAC_Q_sh_lb").value) / 60000;
        const Q_sh_ub = parseFloat(document.getElementById("AAC_Q_sh_ub").value) / 60000;
        const Q_sh_RB = parseFloat(document.getElementById("AAC_Q_sh_RB").value) / 60000;
        const r_1 = parseFloat(document.getElementById("AAC_r_1").value);
        const r_2 = parseFloat(document.getElementById("AAC_r_2").value);
        const L = parseFloat(document.getElementById("AAC_L").value);
        const w_lb = 2 * Math.PI / 60 * parseFloat(document.getElementById("AAC_w_lb").value);
        const w_ub = 2 * Math.PI / 60 * parseFloat(document.getElementById("AAC_w_ub").value);
        const T = parseFloat(document.getElementById("AAC_T").value);
        const P = parseFloat(document.getElementById("AAC_P").value);

        // Validate input parameters
        if (isNaN(Q_a) || isNaN(Q_sh) || isNaN(Q_sh_lb) || isNaN(Q_sh_ub) || isNaN(Q_sh_RB)  || isNaN(r_1) || isNaN(r_2) || isNaN(L) || 
            isNaN(w_lb) || isNaN(w_ub) || isNaN(T) || isNaN(P)) {
            throw new Error('Invalid input parameters');
        }

        console.log('Input parameters:', {
            Q_a: Q_a.toExponential(6),
            Q_sh: Q_sh.toExponential(6),
            Q_sh_lb: Q_sh_lb.toExponential(6),
            Q_sh_ub: Q_sh_ub.toExponential(6),
            Q_sh_RB: Q_sh_RB.toExponential(6),
            r_1: r_1.toExponential(6),
            r_2: r_2.toExponential(6),
            L: L.toExponential(6),
            w_lb: w_lb.toExponential(6),
            w_ub: w_ub.toExponential(6),
            T: T.toExponential(6),
            P: P.toExponential(6)
        });

        // Calculate gas properties
        const mu = 1.81809e-5 * Math.pow(T / 293.15, 1.5) * (293.15 + 110.4) / (T + 110.4);

        // Classifier parameters
        const Q_sh_spa = Array.from({length: 1000}, (_, i) => 
            Math.pow(10, Math.log10(Q_sh_lb) + (Math.log10(Q_sh_ub) - Math.log10(Q_sh_lb)) * i / 999)
        );
        const R_t = Q_sh_spa.map(q => q / Q_a);
        const R_t_lb = Q_sh_lb / Q_a;
        const R_t_ub = Q_sh_ub / Q_a;
        const R_t_i = Q_sh / Q_a;
        const w_low = Q_sh_spa.map(() => w_lb);
        const w_up = Q_sh_spa.map(Q => Q < Q_sh_RB ? Math.min(w_ub, 723.7 - 9.87 * 60000 * Q) : Math.min(w_ub, 875 - 25 * 60000 * Q));

        console.log('Calculation parameters:', {
            mu: mu.toExponential(6),
            Q_sh_spa_range: [Q_sh_spa[0].toExponential(6), Q_sh_spa[Q_sh_spa.length-1].toExponential(6)],
            R_t: [R_t[0].toExponential(6), R_t[Q_sh_spa.length-1].toExponential(6)],
            R_t_lb: R_t_lb.toExponential(6),
            R_t_ub: R_t_ub.toExponential(6),
            R_t_i: R_t_i.toExponential(6),
            w_low: [w_low[0].toExponential(6), w_low[Q_sh_spa.length-1].toExponential(6)],
            w_up: [w_up[0].toExponential(6), w_up[Q_sh_spa.length-1].toExponential(6)]
        });

        // Precompute factors
        const factor1 = w_low.map(w => (36 * mu) / (Math.PI * 1000 * Math.pow(r_1 + r_2, 2) * L * Math.pow(w, 2)));
        const factor2 = w_up.map(w => (36 * mu) / (Math.PI * 1000 * Math.pow(r_1 + r_2, 2) * L * Math.pow(w, 2)));

        console.log('Factors:', {
            factor1: [factor1[0].toExponential(6), factor1[Q_sh_spa.length-1].toExponential(6)],
            factor2: [factor1[0].toExponential(6), factor1[Q_sh_spa.length-1].toExponential(6)]
        });


        // Root-finder (simple Newton-style)
        function findRoot(func, guess, args = [], bounds = [1e-9, 1e-5], tol = 1e-18, maxIter = 1000, h = 1e-8) {
            let x = guess;
            for (let i = 0; i < maxIter; i++) {
                const fx = func(x, ...args);
                const fxh = func(x + h, ...args);
                const dfdx = (fxh - fx) / h;
                if (Math.abs(dfdx) < 1e-20 || !isFinite(dfdx)) break;
                const step = fx / dfdx;
                x = x - step;
                if (x < bounds[0] || x > bounds[1]) return NaN;
                if (Math.abs(fx) < tol) return x;
            }
            return NaN;
        }

        function f(d, Q_sh_val, factor_val) {
            return Math.pow(d, 2) * Cc(d, P, T) - factor_val * Q_sh_val;
        }
    
        const d_min_AAC = Q_sh_spa.map((Q, i) => findRoot(f, 1e-8, [Q, factor2[i]]));
        const d_max_AAC = Q_sh_spa.map((Q, i) => findRoot(f, 1e-5, [Q, factor1[i]]));
    
        const index = Q_sh_spa.reduce((iMin, val, i) => Math.abs(val - Q_sh) < Math.abs(Q_sh_spa[iMin] - Q_sh) ? i : iMin, 0);
        const factor1_input = factor1[index];
        const factor2_input = factor2[index];
    
        const d_i = findRoot(f, 1e-8, [Q_sh, factor2_input]);
        const d_o = findRoot(f, 1e-5, [Q_sh, factor1_input]);

        console.log({
            d_i: d_i?.toExponential?.(6),
            d_o: d_o?.toExponential?.(6)
        });

        document.getElementById('diameter-results_AAC').innerHTML = `
            <strong>Selected Boundaries:</strong><br>
            d_i_AAC = ${(d_i * 1e9).toFixed(2)} nm<br>
            d_o_AAC = ${(d_o * 1e9).toFixed(2)} nm
        `;
                     


        // Plot the results
        const trace1 = {
            x: [d_min_AAC[0] * 1e9, d_max_AAC[0] * 1e9],
            y: [R_t_lb, R_t_lb],
            type: 'scatter',
            mode: 'lines',
            showlegend: false,
            line: {color: 'red'}
        };

        const trace2 = {
            x: [d_min_AAC[d_min_AAC.length-1] * 1e9, d_max_AAC[d_max_AAC.length-1] * 1e9],
            y: [R_t_ub, R_t_ub],
            type: 'scatter',
            mode: 'lines',
            showlegend: false,
            line: {color: 'red'}
        };

        const trace3 = {
            x: [d_i * 1e9, d_o * 1e9],
            y: [R_t_i, R_t_i],
            type: 'scatter',
            mode: 'lines',
            name: 'Input Q_sh boundry',
            showlegend: true,
            line: {color: 'green'}
        };

        const trace4 = {
            x: d_min_AAC.map(d => d * 1e9),
            y: R_t,
            type: 'scatter',
            mode: 'lines',
            showlegend: false,
            line: {color: 'red'}
        };

        const trace5 = {
            x: d_max_AAC.map(d => d * 1e9),
            y: R_t,
            type: 'scatter',
            mode: 'lines',
            showlegend: false,
            line: {color: 'red'}
        };

        const layout = {
            title: 'AAC Operational Range',
            xaxis: {
                title: {
                    text: 'Aerodynamic diameter, Dₐ [nm]',
                    font: { size: 18 },
                    standoff: 15
                },
                type: 'log'
            },
            yaxis: {
                title: {
                    text: 'R<sub>τ</sub>',
                    font: { size: 18 },
                    standoff: 15
                },
                type: 'log'
            },
            grid: {
                rows: 1,
                columns: 1,
                pattern: 'independent'
            }
        };

        // Check if Plotly is loaded
        if (typeof Plotly === 'undefined') {
            throw new Error('Plotly library is not loaded');
        }

        // Clear previous plot
        const plotDiv = document.getElementById('plot_AAC');
        plotDiv.innerHTML = '';

        // Create new plot
        Plotly.newPlot('plot_AAC', [trace1, trace2, trace3, trace4, trace5], layout)
            .then(() => console.log('Plot created successfully'))
            .catch(err => console.error('Error creating plot:', err));

    } catch (error) {
        console.error('Error in calculateCPMARange:', error);
        alert('An error occurred while calculating the CPMA range. Please check the console for details.');
    }

}

// Add event listener for when the page loads
document.addEventListener('DOMContentLoaded', function() {
    // Check if Plotly is loaded
    if (typeof Plotly === 'undefined') {
        console.error('Plotly library is not loaded');
        alert('Error: Plotly library is not loaded. Please check your internet connection.');
    }
}); 
document.addEventListener('DOMContentLoaded', () => {
    const electroModel = document.getElementById("Electrostatic-model");
    const dmaModel = document.getElementById("DMA-model");

    const inputs = {
        DMA_Q_a: document.getElementById("DMA_Q_a"),
        DMA_Q_sh: document.getElementById("DMA_Q_sh"),
        DMA_Q_sh_lb: document.getElementById("DMA_Q_sh_lb"),
        DMA_Q_sh_ub: document.getElementById("DMA_Q_sh_ub"),
        DMA_r_1: document.getElementById("DMA_r_1"),
        DMA_r_2: document.getElementById("DMA_r_2"),
        DMA_L: document.getElementById("DMA_L"),
        DMA_V_min: document.getElementById("DMA_V_min"),
        DMA_V_max: document.getElementById("DMA_V_max"),
        DMA_T: document.getElementById("DMA_T"),
        DMA_P: document.getElementById("DMA_P"),
    };

    const config = {
        "3080": {
            DMA_Q_sh_lb: 2,
            DMA_Q_sh_ub: 15,
            DMA_V_min: 10,
            DMA_V_max: 10000,
            DMA_T: 298.15,
            DMA_P: 101325
        },
        "3082": {
            DMA_Q_sh_lb: 2,
            DMA_Q_sh_ub: 30,
            DMA_V_min: 10,
            DMA_V_max: 10000,
            DMA_T: 298.15,
            DMA_P: 101325
        },
        "3081-long": {
            DMA_r_1: 0.00937,
            DMA_r_2: 0.01961,
            DMA_L: 0.44369
        },
        "3081A-long": {
            DMA_r_1: 0.00937,
            DMA_r_2: 0.01961,
            DMA_L: 0.44369
        },
        "3085-nano-single": {
            DMA_Q_sh_ub: 15,
            DMA_r_1: 0.00937,
            DMA_r_2: 0.01905,
            DMA_L: 0.04987
        },
        "3085-nano-dual": {
            DMA_Q_sh_ub: 20,
            DMA_r_1: 0.00937,
            DMA_r_2: 0.01905,
            DMA_L: 0.04987
        },
        "3085A-nano": {
            DMA_r_1: 0.00937,
            DMA_r_2: 0.01905,
            DMA_L: 0.04987
        },
        "3086-1nm": {
            DMA_r_1: 0.00937,
            DMA_r_2: 0.01905,
            DMA_L: 0.02
        },
    };

    function updateInputs() {
        const eVal = electroModel.value;
        const dVal = dmaModel.value;

        // Clear all fields first
        Object.values(inputs).forEach(input => input.disabled = false);

        // Apply electrostatic model values if defined
        if (eVal !== "non" && config[eVal]) {
            for (const [key, val] of Object.entries(config[eVal])) {
                inputs[key].value = val;
            }
        }

        // Apply DMA model values if defined
        if (dVal !== "non" && config[dVal]) {
            for (const [key, val] of Object.entries(config[dVal])) {
                inputs[key].value = val;
            }
        }
    }

    // Listen for changes
    electroModel.addEventListener("change", updateInputs);
    dmaModel.addEventListener("change", updateInputs);
});

document.addEventListener('DOMContentLoaded', () => {
    const CPMAModel = document.getElementById("CPMA-model");

    const inputs = {
        w_lb: document.getElementById("w_lb")

    };

    const config = {
        "Mk1": {
            w_lb: 500,
        },
        "Mk2": {
            w_lb: 200,
        }
    };
        function updateInputs() {
            const cVal = CPMAModel.value;
    
            // Clear all fields first
            Object.values(inputs).forEach(input => input.disabled = false);
    
            // Apply CPMA model values if defined
            if (cVal !== "non" && config[cVal]) {
                for (const [key, val] of Object.entries(config[cVal])) {
                    inputs[key].value = val;
                }
            }
        }
    
        // Listen for changes
        CPMAModel.addEventListener("change", updateInputs);
    });
