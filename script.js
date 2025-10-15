// Constants
const e = 1.6e-19;

// Global data storage for tandem comparison
let globalClassifierData = {
    DMA: null,
    AAC: null,
    CPMA: null
};

// Function to convert aerodynamic diameter to mobility diameter for AAC
function da_rhoeff2dm(da, rho100, Dm, P, T) {
    // Convert aerodynamic diameter and effective density to mobility diameter

    const rho0 = 1e3; // Density of water [kg/m^3]
    const k = Math.PI / 6 * rho100 * Math.pow(100 * 1e-9, 3 - Dm);

    // Initial guess: direct method (no iteration)
    let dm_in = da * Math.sqrt(rho0 / rho100);

    // Iterative function for root-finding
    function fun_iter(dm) {
        return  (
            dm  * Math.sqrt(
                (6 * k / Math.PI * Math.pow(dm, Dm - 3) / rho0) *
                (Cc(dm, P, T) / Cc(da, P, T))
            ) - da
        );
    }

    function solve(func, x0, tol = 1e-9, maxIter = 100) {
        let x = x0;
        for (let i = 0; i < maxIter; i++) {
            const fx = func(x);
            const h = 1e-8;
            const dfx = (func(x + h) - func(x - h)) / (2 * h); // derivative approx
            if (Math.abs(dfx) < 1e-12) break;
            const xNew = x - fx / dfx;
            if (Math.abs(xNew - x) < tol) return xNew;
            x = xNew;
        }
        return x;
    }

    let dm = solve(fun_iter, dm_in ) ;

    return dm; 
}

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

        // Store data for tandem comparison
        let x_data;
        if (x_label === 'Mass [fg]') {
            x_data = x_vals.map(m => Math.pow((m * 1e-18) / k, 1 / Dm)* 1e9);
        } else {
            x_data = x_vals; // Already in correct units
        }

        globalClassifierData.CPMA = {
            type: 'CPMA',
            xLabel: 'Mobility diameter, Dₘ [nm]',
            x: x_data,
            y_min: R_m_1, // Lower boundary
            y_max: R_m_2, // Upper boundary
            yLabel: 'Rₘ',
            color: 'green',
            name: 'CPMA',
            d_i: parseFloat(d_i_label), // Store inner diameter
            d_o: parseFloat(d_o_label), // Store outer diameter
            Dm: Dm, // Store mass-mobility exponent for tandem plotting
            isCPMA: true // Flag to identify CPMA in tandem plotting
        };
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

        // Store data for tandem comparison
        globalClassifierData.DMA = {
            type: 'DMA',
            x_min: d_min_DMA.map(d => d * 1e9), // Convert to nm
            x_max: d_max_DMA.map(d => d * 1e9), // Convert to nm
            y: R_B,
            xLabel: 'Mobility diameter, Dₘ [nm]',
            yLabel: 'Rₘ',
            color: 'blue',
            name: 'DMA',
            R_lb: R_B_lb, 
            R_ub: R_B_ub  
        };

    } catch (error) {
        console.error('Error in calculateDMARange:', error);
        alert('An error occurred while calculating the DMA range. Please check the console for details.');
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

        // Get current CPMA values for defaults
        const cpmRho100 = parseFloat(document.getElementById('rho100').value);
        const cpmDm = parseFloat(document.getElementById('Dm').value);
        
        // Store data for tandem comparison
        globalClassifierData.AAC = {
            type: 'AAC',
            x_min: d_min_AAC.map(d => d * 1e9), // Convert to nm
            x_max: d_max_AAC.map(d => d * 1e9), // Convert to nm
            y: R_t,
            xLabel: 'Aerodynamic diameter, Dₐ [nm]',
            yLabel: 'Rτ',
            color: 'red',
            name: 'AAC',
            R_lb: R_t_lb, 
            R_ub: R_t_ub,
            isAAC: true, // Flag to identify AAC in tandem plotting
            // Store parameters needed for da to dm conversion
            rho100: cpmRho100, // Use CPMA effective density as default
            Dm: cpmDm, // Use CPMA mass-mobility exponent as default
            P: P, // Pressure
            T: T  // Temperature
        };

    } catch (error) {
        console.error('Error in calculateAACRange:', error);
        alert('An error occurred while calculating the AAC range. Please check the console for details.');
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

// Sync tandem parameters with CPMA section
document.addEventListener('DOMContentLoaded', () => {
    const tandemRho100 = document.getElementById('tandem-rho100');
    const tandemDm = document.getElementById('tandem-Dm');
    const cpmRho100 = document.getElementById('rho100');
    const cpmDm = document.getElementById('Dm');

    const classifier1Select = document.getElementById('tandem-classifier1');
    const classifier2Select = document.getElementById('tandem-classifier2');

    // Function to sync tandem values with CPMA values
    function syncTandemValues() {
        // Check if either classifier is CPMA
        const classifier1 = classifier1Select.value;
        const classifier2 = classifier2Select.value;

        if (classifier1 === "CPMA" || classifier2 === "CPMA") {
            // Disable user editing
            tandemRho100.disabled = true;
            tandemDm.disabled = true;

            // Sync CPMA values
            if (cpmRho100) tandemRho100.value = cpmRho100.value;
            if (cpmDm) tandemDm.value = cpmDm.value;
        } else {
            // Re-enable if CPMA not involved
            tandemRho100.disabled = false;
            tandemDm.disabled = false;
        }
    }

    // Sync when CPMA values change
    if (cpmRho100) {
        cpmRho100.addEventListener('input', syncTandemValues);
    }
    if (cpmDm) {
        cpmDm.addEventListener('input', syncTandemValues);
    }
    if (classifier1Select) {
        classifier1Select.addEventListener('change', syncTandemValues);
    }
    if (classifier2Select) {
        classifier2Select.addEventListener('change', syncTandemValues);
    }

    // Initial sync
    syncTandemValues();
});

// Tandem Classifier Comparison Functions
function calculateTandemComparison() {
    try {
        const classifier1 = document.getElementById('tandem-classifier1').value;
        const classifier2 = document.getElementById('tandem-classifier2').value;
        const plotMode = document.getElementById('tandem-plot-mode').value;
        
        // Get tandem parameters
        const tandemRho100 = parseFloat(document.getElementById('tandem-rho100').value);
        const tandemDm = parseFloat(document.getElementById('tandem-Dm').value);
        
        if (classifier1 === classifier2) {
            alert('Please select two different classifiers for comparison.');
            return;
        }
        
        // Get data for both classifiers
        let data1 = getClassifierData(classifier1);
        let data2 = getClassifierData(classifier2);
        
        if (!data1) {
            alert(`Error: No data available for ${classifier1}. Please calculate ${classifier1} first.`);
            return;
        }
        
        if (!data2) {
            alert(`Error: No data available for ${classifier2}. Please calculate ${classifier2} first.`);
            return;
        }
        
        // Check if data arrays are empty
        const data1Valid = (data1.x_min && data1.x_min.length > 0) || (data1.x && data1.x.length > 0);
        const data2Valid = (data2.x_min && data2.x_min.length > 0) || (data2.x && data2.x.length > 0);
        
        if (!data1Valid) {
            alert(`Error: ${classifier1} data is empty. Please recalculate ${classifier1}.`);
            return;
        }
        
        if (!data2Valid) {
            alert(`Error: ${classifier2} data is empty. Please recalculate ${classifier2}.`);
            return;
        }
        
        // Update AAC data with tandem parameters if needed
        if (data1.isAAC) {
            data1 = { ...data1, rho100: tandemRho100, Dm: tandemDm };
        }
        if (data2.isAAC) {
            data2 = { ...data2, rho100: tandemRho100, Dm: tandemDm };
        }
        
        // Create comparison plot
        createTandemPlot(data1, data2, classifier1, classifier2, plotMode);
        
        // Display results
        displayTandemResults(data1, data2, classifier1, classifier2);
        
    } catch (error) {
        console.error('Error in calculateTandemComparison:', error);
        alert('An error occurred while comparing classifiers. Please check the console for details.');
    }
}

function getClassifierData(classifierType) {
    // Return stored data if available, otherwise return null
    return globalClassifierData[classifierType];
}

function createTandemPlot(data1, data2, classifier1, classifier2, plotMode) {
    // Clear previous plot
    const plotDiv = document.getElementById('plot_tandem');
    plotDiv.innerHTML = '';
    
    if (plotMode === 'overlay') {
        createOverlayPlot(data1, data2, classifier1, classifier2);
    } else {
        createSideBySidePlot(data1, data2, classifier1, classifier2);
    }
}

function createOverlayPlot(data1, data2, classifier1, classifier2) {
    const traces = [];
    
    // Add traces for first classifier (both boundaries)
    if (data1.x_min && data1.x_min.length > 0) {
        // DMA/AAC case: x_min and x_max with same y values
        if (data1.isAAC) {
            // For AAC in tandem: convert da to dm
            const x_min_dm = data1.x_min.map(da => da_rhoeff2dm(da * 1e-9, data1.rho100, data1.Dm, data1.P, data1.T) * 1e9);
            const x_max_dm = data1.x_max.map(da => da_rhoeff2dm(da * 1e-9, data1.rho100, data1.Dm, data1.P, data1.T) * 1e9);
            
            traces.push({
                x: x_min_dm,
                y: data1.y,
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name}`,
                line: { color: data1.color, dash: 'solid' },
                showlegend: true
            });
            traces.push({
                x: x_max_dm,
                y: data1.y,
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name}`,
                line: { color: data1.color, dash: 'solid' },
                showlegend: false
            });
            traces.push({
                x: [x_min_dm[0], x_max_dm[0]],
                y: [data1.R_lb, data1.R_lb],
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name}`,
                line: { color: data1.color, dash: 'solid' },
                showlegend: false
            });
            traces.push({
                x: [x_min_dm[x_min_dm.length-1], x_max_dm[x_min_dm.length-1]],
                y: [data1.R_ub, data1.R_ub],
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name}`,
                line: { color: data1.color, dash: 'solid' },
                showlegend: false
            });
        } else {
            // Regular DMA case
            traces.push({
                x: data1.x_min,
                y: data1.y,
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name}`,
                line: { color: data1.color, dash: 'solid' },
                showlegend: true
            });
            traces.push({
                x: data1.x_max,
                y: data1.y,
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name}`,
                line: { color: data1.color, dash: 'solid' },
                showlegend: false
            });
            traces.push({
                x: [data1.x_min[0], data1.x_max[0]],
                y: [data1.R_lb, data1.R_lb],
                type: 'scatter',
                mode: 'lines',
                line: { color: data1.color, dash: 'solid' },
                showlegend: false
            });
            traces.push({
                x: [data1.x_min[data1.x_min.length-1], data1.x_max[data1.x_min.length-1]],
                y: [data1.R_ub, data1.R_ub],
                type: 'scatter',
                mode: 'lines',
                line: { color: data1.color, dash: 'solid' },
                showlegend: false
            });
        }
    } else if (data1.x && data1.x.length > 0) {
        // CPMA case: same x values with y_min and y_max
        if (data1.isCPMA) {
            // For CPMA in tandem: use Dm * R_m on y-axis
            traces.push({
                x: data1.x,
                y: data1.y_min.map(rm => data1.Dm * rm),
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name}`,
                line: { color: data1.color, dash: 'solid' },
                showlegend: true
            });
            traces.push({
                x: data1.x,
                y: data1.y_max.map(rm => data1.Dm * rm),
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name}`,
                line: { color: data1.color, dash: 'solid' },
                showlegend: false
            });
        } else {
            // Regular case for other classifiers
            traces.push({
                x: data1.x,
                y: data1.y_min,
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name}`,
                line: { color: data1.color, dash: 'solid' },
                showlegend: true
            });
            traces.push({
                x: data1.x,
                y: data1.y_max,
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name}`,
                line: { color: data1.color, dash: 'solid' },
                showlegend: false
            });
            traces.push({
                x: [data1.x_min[0], data1.x_max[0]],
                y: [data1.R_lb, data1.R_lb],
                type: 'scatter',
                mode: 'lines',
                line: { color: data1.color, dash: 'solid' },
                showlegend: false
            });
            traces.push({
                x: [data1.x_min[data1.x_min.length-1], data1.x_max[data1.x_min.length-1]],
                y: [data1.R_ub, data1.R_ub],
                type: 'scatter',
                mode: 'lines',
                line: { color: data1.color, dash: 'solid' },
                showlegend: false
            });
        }
    }
    
    // Add traces for second classifier (both boundaries)
    if (data2.x_min && data2.x_min.length > 0) {
        // DMA/AAC case: x_min and x_max with same y values
        if (data2.isAAC) {
            // For AAC in tandem: convert da to dm
            const x_min_dm = data2.x_min.map(da => da_rhoeff2dm(da * 1e-9, data2.rho100, data2.Dm, data2.P, data2.T) * 1e9);
            const x_max_dm = data2.x_max.map(da => da_rhoeff2dm(da * 1e-9, data2.rho100, data2.Dm, data2.P, data2.T) * 1e9);
            
            traces.push({
                x: x_min_dm,
                y: data2.y,
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name}`,
                line: { color: data2.color, dash: 'dash' },
                showlegend: true
            });
            traces.push({
                x: x_max_dm,
                y: data2.y,
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name}`,
                line: { color: data2.color, dash: 'dash' },
                showlegend: false
            });
            traces.push({
                x: [x_min_dm[0], x_max_dm[0]],
                y: [data2.R_lb, data2.R_lb],
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name}`,
                line: { color: data2.color, dash: 'dash' },
                showlegend: false
            });
            traces.push({
                x: [x_min_dm[x_min_dm.length-1], x_max_dm[x_min_dm.length-1]],
                y: [data2.R_ub, data2.R_ub],
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name}`,
                line: { color: data2.color, dash: 'dash' },
                showlegend: false
            });
        } else {
            // Regular DMA case
            traces.push({
                x: data2.x_min,
                y: data2.y,
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name}`,
                line: { color: data2.color, dash: 'dash' },
                showlegend: true
            });
            traces.push({
                x: data2.x_max,
                y: data2.y,
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name}`,
                line: { color: data2.color, dash: 'dash' },
                showlegend: false
            });
            traces.push({
                x: [data2.x_min[0], data2.x_max[0]],
                y: [data2.R_lb, data2.R_lb],
                type: 'scatter',
                mode: 'lines',
                line: { color: data2.color, dash: 'dash' },
                showlegend: false
            });
            traces.push({
                x: [data2.x_min[data2.x_min.length-1], data2.x_max[data2.x_min.length-1]],
                y: [data2.R_ub, data2.R_ub],
                type: 'scatter',
                mode: 'lines',
                line: { color: data2.color, dash: 'dash' },
                showlegend: false
            });
        }
    } else if (data2.x && data2.x.length > 0) {
        // CPMA case: same x values with y_min and y_max
        if (data2.isCPMA) {
            // For CPMA in tandem: use Dm * R_m on y-axis
            traces.push({
                x: data2.x,
                y: data2.y_min.map(rm => data2.Dm * rm),
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name}`,
                line: { color: data2.color, dash: 'dash' },
                showlegend: true
            });
            traces.push({
                x: data2.x,
                y: data2.y_max.map(rm => data2.Dm * rm),
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name}`,
                line: { color: data2.color, dash: 'dash' },
                showlegend: false
            });
        } else {
            // Regular case for other classifiers
            traces.push({
                x: data2.x,
                y: data2.y_min,
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name}`,
                line: { color: data2.color, dash: 'dash' },
                showlegend: true
            });
            traces.push({
                x: data2.x,
                y: data2.y_max,
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name}`,
                line: { color: data2.color, dash: 'dash' },
                showlegend: false
            });
            traces.push({
                x: [data2.x_min[0], data2.x_max[0]],
                y: [data2.R_lb, data2.R_lb],
                type: 'scatter',
                mode: 'lines',
                line: { color: data2.color, dash: 'dash' },
                showlegend: false
            });
            traces.push({
                x: [data2.x_min[data2.x_min.length-1], data2.x_max[data2.x_min.length-1]],
                y: [data2.R_ub, data2.R_ub],
                type: 'scatter',
                mode: 'lines',
                line: { color: data2.color, dash: 'dash' },
                showlegend: false
            });
        }
    }
    
    // Determine axis labels based on which classifiers are involved
    let yAxisLabel = 'Resolution Parameter';
    let xAxisLabel = 'Mobility diameter, Dₘ [nm]';
    
    if (data1.isCPMA || data2.isCPMA) {
        yAxisLabel = 'Dm × Rₘ';
    }
    
    // If AAC is involved, x-axis should be mobility diameter
    if (data1.isAAC || data2.isAAC) {
        xAxisLabel = 'Mobility diameter, Dₘ [nm]';
    }
    
    const layout = {
        title: `Tandem ${classifier1} -- ${classifier2}`,
        xaxis: {
            title: {
                text: xAxisLabel,
                font: { size: 18 },
                standoff: 15
            },
            type: 'log'
        },
        yaxis: {
            title: {
                text: yAxisLabel,
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
    
    Plotly.newPlot('plot_tandem', traces, layout)
        .then(() => console.log('Tandem overlay plot created successfully'))
        .catch(err => console.error('Error creating tandem plot:', err));
}

function createSideBySidePlot(data1, data2, classifier1, classifier2) {
    const traces = [];
    
    // First classifier (left subplot) - both boundaries
    if (data1.x_min && data1.x_min.length > 0) {
        // DMA/AAC case: x_min and x_max with same y values
        if (data1.isAAC) {
            // For AAC in tandem: convert da to dm
            const x_min_dm = data1.x_min.map(da => da_rhoeff2dm(da * 1e-9, data1.rho100, data1.Dm, data1.P, data1.T) * 1e9);
            const x_max_dm = data1.x_max.map(da => da_rhoeff2dm(da * 1e-9, data1.rho100, data1.Dm, data1.P, data1.T) * 1e9);
            
            traces.push({
                x: x_min_dm,
                y: data1.y,
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name} (min)`,
                line: { color: data1.color, dash: 'solid' },
                xaxis: 'x1',
                yaxis: 'y1',
                showlegend: true
            });
            traces.push({
                x: x_max_dm,
                y: data1.y,
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name} (max)`,
                line: { color: data1.color, dash: 'solid' },
                xaxis: 'x1',
                yaxis: 'y1',
                showlegend: false
            });
            traces.push({
                x: [x_min_dm[0], x_max_dm[0]],
                y: [data1.R_lb, data1.R_lb],
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name} (max)`,
                line: { color: data1.color, dash: 'solid' },
                xaxis: 'x1',
                yaxis: 'y1',
                showlegend: false
            });
            traces.push({
                x: [x_min_dm[x_min_dm.length-1], x_max_dm[x_min_dm.length-1]],
                y: [data1.R_ub, data1.R_ub],
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name} (max)`,
                line: { color: data1.color, dash: 'solid' },
                xaxis: 'x1',
                yaxis: 'y1',
                showlegend: false
            }); 
        } else {
            // Regular DMA case
            traces.push({
                x: data1.x_min,
                y: data1.y,
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name} (min)`,
                line: { color: data1.color, dash: 'solid' },
                xaxis: 'x1',
                yaxis: 'y1',
                showlegend: true
            });
            traces.push({
                x: data1.x_max,
                y: data1.y,
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name} (max)`,
                line: { color: data1.color, dash: 'solid' },
                xaxis: 'x1',
                yaxis: 'y1',
                showlegend: true
            });
        }
    } else if (data1.x && data1.x.length > 0) {
        // CPMA case: same x values with y_min and y_max
        if (data1.isCPMA) {
            // For CPMA in tandem: use Dm * R_m on y-axis
            traces.push({
                x: data1.x,
                y: data1.y_min.map(rm => data1.Dm * rm),
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name} (min)`,
                line: { color: data1.color, dash: 'solid' },
                xaxis: 'x1',
                yaxis: 'y1',
                showlegend: true
            });
            traces.push({
                x: data1.x,
                y: data1.y_max.map(rm => data1.Dm * rm),
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name} (max)`,
                line: { color: data1.color, dash: 'solid' },
                xaxis: 'x1',
                yaxis: 'y1',
                showlegend: true
            });
        } else {
            // Regular case for other classifiers
            traces.push({
                x: data1.x,
                y: data1.y_min,
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name} (min)`,
                line: { color: data1.color, dash: 'solid' },
                xaxis: 'x1',
                yaxis: 'y1',
                showlegend: true
            });
            traces.push({
                x: data1.x,
                y: data1.y_max,
                type: 'scatter',
                mode: 'lines',
                name: `${data1.name} (max)`,
                line: { color: data1.color, dash: 'solid' },
                xaxis: 'x1',
                yaxis: 'y1',
                showlegend: true
            });
        }
    }
    
    // Second classifier (right subplot) - both boundaries
    if (data2.x_min && data2.x_min.length > 0) {
        // DMA/AAC case: x_min and x_max with same y values
        if (data2.isAAC) {
            // For AAC in tandem: convert da to dm
            const x_min_dm = data2.x_min.map(da => da_rhoeff2dm(da * 1e-9, data2.rho100, data2.Dm, data2.P, data2.T) * 1e9);
            const x_max_dm = data2.x_max.map(da => da_rhoeff2dm(da * 1e-9, data2.rho100, data2.Dm, data2.P, data2.T) * 1e9);
            
            traces.push({
                x: x_min_dm,
                y: data2.y,
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name} (min)`,
                line: { color: data2.color, dash: 'dash' },
                xaxis: 'x2',
                yaxis: 'y2',
                showlegend: true
            });
            traces.push({
                x: x_max_dm,
                y: data2.y,
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name} (max)`,
                line: { color: data2.color, dash: 'dash' },
                xaxis: 'x2',
                yaxis: 'y2',
                showlegend: true
            });
            traces.push({
                x: [x_min_dm[0], x_max_dm[0]],
                y: [data2.R_lb, data2.R_lb],
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name} (max)`,
                line: { color: data2.color, dash: 'solid' },
                xaxis: 'x2',
                yaxis: 'y2',
                showlegend: false
            });
            traces.push({
                x: [x_min_dm[x_min_dm.length-1], x_max_dm[x_min_dm.length-1]],
                y: [data2.R_ub, data2.R_ub],
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name} (max)`,
                line: { color: data2.color, dash: 'solid' },
                xaxis: 'x2',
                yaxis: 'y2',
                showlegend: false
            });
        } else {
            // Regular DMA case
            traces.push({
                x: data2.x_min,
                y: data2.y,
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name} (min)`,
                line: { color: data2.color, dash: 'dash' },
                xaxis: 'x2',
                yaxis: 'y2',
                showlegend: true
            });
            traces.push({
                x: data2.x_max,
                y: data2.y,
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name} (max)`,
                line: { color: data2.color, dash: 'dash' },
                xaxis: 'x2',
                yaxis: 'y2',
                showlegend: true
            });
        }
    } else if (data2.x && data2.x.length > 0) {
        // CPMA case: same x values with y_min and y_max
        if (data2.isCPMA) {
            // For CPMA in tandem: use Dm * R_m on y-axis
            traces.push({
                x: data2.x,
                y: data2.y_min.map(rm => data2.Dm * rm),
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name} (min)`,
                line: { color: data2.color, dash: 'dash' },
                xaxis: 'x2',
                yaxis: 'y2',
                showlegend: true
            });
            traces.push({
                x: data2.x,
                y: data2.y_max.map(rm => data2.Dm * rm),
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name} (max)`,
                line: { color: data2.color, dash: 'dash' },
                xaxis: 'x2',
                yaxis: 'y2',
                showlegend: true
            });
        } else {
            // Regular case for other classifiers
            traces.push({
                x: data2.x,
                y: data2.y_min,
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name} (min)`,
                line: { color: data2.color, dash: 'dash' },
                xaxis: 'x2',
                yaxis: 'y2',
                showlegend: true
            });
            traces.push({
                x: data2.x,
                y: data2.y_max,
                type: 'scatter',
                mode: 'lines',
                name: `${data2.name} (max)`,
                line: { color: data2.color, dash: 'dash' },
                xaxis: 'x2',
                yaxis: 'y2',
                showlegend: true
            });
        }
    }
    
    // Determine axis labels based on which classifiers are involved
    let y1Label = data1.yLabel;
    let y2Label = data2.yLabel;
    let x1Label = data1.xLabel;
    let x2Label = data2.xLabel;
    
    if (data1.isCPMA) {
        y1Label = 'Dm × Rₘ';
    }
    if (data2.isCPMA) {
        y2Label = 'Dm × Rₘ';
    }
    
    // If AAC is involved, x-axis should be mobility diameter
    if (data1.isAAC) {
        x1Label = 'Mobility diameter, Dₘ [nm]';
    }
    if (data2.isAAC) {
        x2Label = 'Mobility diameter, Dₘ [nm]';
    }
    
    const layout = {
        title: `Tandem ${classifier1} -- ${classifier2}`,
        xaxis: {
            title: {
                text: x1Label,
                font: { size: 16 }
            },
            type: 'log',
            domain: [0, 0.45]
        },
        yaxis: {
            title: {
                text: y1Label,
                font: { size: 16 }
            },
            type: 'log'
        },
        xaxis2: {
            title: {
                text: x2Label,
                font: { size: 16 }
            },
            type: 'log',
            domain: [0.55, 1]
        },
        yaxis2: {
            title: {
                text: y2Label, 
                font: { size: 16 }
            },
            type: 'log',
            anchor: 'x2'
        },
        grid: {
            rows: 1,
            columns: 2,
            pattern: 'independent'
        }
    };
    
    Plotly.newPlot('plot_tandem', traces, layout)
        .then(() => console.log('Tandem side-by-side plot created successfully'))
        .catch(err => console.error('Error creating tandem plot:', err));
}

function displayTandemResults(data1, data2, classifier1, classifier2) {
    const resultsDiv = document.getElementById('tandem-results');
    
    let html = `<strong>Comparison Results:</strong><br><br>`;
    
    // First classifier results
    html += `<strong>${classifier1}:</strong><br>`;
    
    // Show correct x-axis label
    const x1Label = data1.isAAC ? 'Mobility diameter, Dₘ [nm]' : data1.xLabel;
    html += `- X-axis: ${x1Label}<br>`;
    
    // Show correct y-axis label for CPMA
    const y1Label = data1.isCPMA ? 'Dm × Rₘ' : data1.yLabel;
    html += `- Y-axis: ${y1Label}<br>`;
    
    
    // Show Dm value for CPMA
    if (data1.isCPMA && data1.Dm !== undefined) {
        html += `- Mass-Mobility Exponent (Dm): ${data1.Dm}<br>`;
    }
    
    // Show conversion info for AAC
    if (data1.isAAC) {
        html += `- Converted from aerodynamic to mobility diameter<br>`;
        html += `- Effective density (ρ₁₀₀): ${data1.rho100} kg/m³<br>`;
        html += `- Mass-Mobility Exponent (Dm): ${data1.Dm}<br>`;
    }
    html += `<br>`;
    
    // Second classifier results
    html += `<strong>${classifier2}:</strong><br>`;
    
    // Show correct x-axis label
    const x2Label = data2.isAAC ? 'Mobility diameter, Dₘ [nm]' : data2.xLabel;
    html += `- X-axis: ${x2Label}<br>`;
    
    // Show correct y-axis label for CPMA
    const y2Label = data2.isCPMA ? 'Dm × Rₘ' : data2.yLabel;
    html += `- Y-axis: ${y2Label}<br>`;
    
    
    // Show Dm value for CPMA
    if (data2.isCPMA && data2.Dm !== undefined) {
        html += `- Mass-Mobility Exponent (Dm): ${data2.Dm}<br>`;
    }
    
    // Show conversion info for AAC
    if (data2.isAAC) {
        html += `- Converted from aerodynamic to mobility diameter<br>`;
        html += `- Effective density (ρ₁₀₀): ${data2.rho100} kg/m³<br>`;
        html += `- Mass-Mobility Exponent (Dm): ${data2.Dm}<br>`;
    }
    
    resultsDiv.innerHTML = html;
}