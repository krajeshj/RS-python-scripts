const fs = require('fs');

// Load real production data
const stockHist = JSON.parse(fs.readFileSync('output/stock_history.json', 'utf8'));
const webData = JSON.parse(fs.readFileSync('output/web_data.json', 'utf8'));

// 1. Get the Top 40 Leaders (as the live dashboard would)
let allStocks = [];
if(webData.stocks) allStocks = allStocks.concat(webData.stocks);
if(webData.pulse) allStocks = allStocks.concat(webData.pulse);
if(webData.all_stocks) allStocks = allStocks.concat(webData.all_stocks);

// Unique and sorted by RS
let unq = {};
allStocks.forEach(s => { if(!unq[s.ticker] || (s.rs > unq[s.ticker].rs)) unq[s.ticker] = s; });
let sorted = Object.keys(unq).map(k => unq[k]).sort((a,b) => b.rs - a.rs);
let top40 = sorted.slice(0, 40);

console.log("Top 40 Tickers identified.");

function stdDev(arr) {
    if(!arr.length) return 0;
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    return Math.sqrt(arr.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / arr.length);
}

// ROBUST RRG LOGIC (The Proposed Fix)
function getRobustCoords(s, history) {
    // Falls back to current RS if history is missing
    let curRS = (history && history.length) ? history[0] : s.rs;
    
    // NEW SCIENTIFIC ENGINE
    let h = [...(history || [])].reverse();
    let raw = h.slice(0, 25);
    
    // Fallback if history is too thin for RRG math
    if (raw.length < 5) {
        return {
            ticker: s.ticker,
            x: curRS,
            y: 0, // No momentum fallback
            trail: [{ x: curRS, y: 0 }] 
        };
    }

    // JdK-style smoothing (Same as scientifically overhauled logic)
    const sms = (arr, n) => arr.map((_, i) => i < n-1 ? null : arr.slice(i-n+1, i+1).reduce((a,b)=>a+b,0)/n);
    let rs_ratio = sms(raw, 5);
    let rs_mom = rs_ratio.map((v, i, arr) => {
        if (i < 5 || v === null || arr[i-5] === null) return null;
        return (v / arr[i-5] - 1) * 400;
    });
    let rs_mom_smooth = sms(rs_mom.map(v => v === null ? 0 : v), 3);

    let trail = [];
    for (let i = Math.max(0, rs_ratio.length - 12); i < rs_ratio.length; i++) {
        if (rs_ratio[i] === null) continue;
        trail.push({ x: rs_ratio[i], y: rs_mom_smooth[i] || 0 });
    }

    let last = trail[trail.length - 1];
    return {
        ticker: s.ticker,
        x: last.x,
        y: last.y,
        trail: trail
    };
}

// Execution
const results = top40.map(s => getRobustCoords(s, stockHist[s.ticker]));

const allX = results.flatMap(r => r.trail.map(c => c.x));
const allY = results.flatMap(r => r.trail.map(c => c.y));

console.log("--- ROBUSTNESS TEST RESULTS ---");
console.log("Visible Tickers:", results.length, "/ 40");
console.log("Tickers with Curves:", results.filter(r => r.trail.length > 1).length);
console.log("Tickers as Dots:", results.filter(r => r.trail.length === 1).length);

console.log("X-Std:", stdDev(allX).toFixed(2));
console.log("Y-Std:", stdDev(allY).toFixed(2));

// SVG Mocking for Visual Proof
let svg = `<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg" style="background:#0b0e14">`;
const CW=800, CH=600, P=50;

const meanX = allX.reduce((a,b)=>a+b,0)/allX.length;
const meanY = allY.reduce((a,b)=>a+b,0)/allY.length;
const sX = stdDev(allX) || 10;
const sY = stdDev(allY) || 10;
const rX = sX * 2.5; const rY = sY * 2.5;

const tX = x => P + ((x - (meanX - rX)) / (rX * 2)) * (CW - P * 2);
const tY = y => (CH - P) - ((y - (meanY - rY)) / (rY * 2)) * (CH - P * 2);

results.forEach(r => {
    if(r.trail.length > 1) {
        let path = `M ${tX(r.trail[0].x)} ${tY(r.trail[0].y)}`;
        r.trail.slice(1).forEach(p => path += ` L ${tX(p.x)} ${tY(p.y)}`);
        svg += `<path d="${path}" stroke="cyan" fill="none" opacity="0.4" stroke-width="2"/>`;
    }
    svg += `<circle cx="${tX(r.x)}" cy="${tY(r.y)}" r="4" fill="white" />`;
    svg += `<text x="${tX(r.x)+5}" y="${tY(r.y)}" fill="#94a3b8" font-size="10">${r.ticker}</text>`;
});

svg += `</svg>`;
fs.writeFileSync('scratch/robust_test_proof.svg', svg);
console.log("Verification SVG saved to scratch/robust_test_proof.svg");
