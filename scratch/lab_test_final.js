const fs = require('fs');
const path = require('path');

// Mocking the environment
const stockHist = JSON.parse(fs.readFileSync('output/stock_history.json', 'utf8'));
const webData = JSON.parse(fs.readFileSync('output/web_data.json', 'utf8'));

// Sample Tickers from the "Nope" screenshot or general leaders
const tickers = ["ADI", "KEYS", "HSHP", "CAT", "ZIM", "TRMD"]; 

function median(arr) {
    var s = arr.slice().sort((a,b) => a-b);
    if(!s.length) return 0;
    var m = Math.floor(s.length/2);
    return s.length % 2 ? s[m] : (s[m-1] + s[m])/2;
}

function std(arr) {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const sqDiffs = arr.map(v => Math.pow(v - mean, 2));
    return Math.sqrt(sqDiffs.reduce((a, b) => a + b, 0) / arr.length);
}

// JdK-inspired Logic (Simplified for JS implementation)
function getRRGCoords(ticker, history) {
    // 1. Extract raw RS
    let raw = history.slice(0, 25).reverse(); 
    if (raw.length < 15) return null;

    // 2. RS-Ratio (Smoothed RS) - SMA 5 (for more responsiveness)
    const sms = (arr, n) => arr.map((_, i) => i < n-1 ? null : arr.slice(i-n+1, i+1).reduce((a,b)=>a+b,0)/n);
    let rs_ratio = sms(raw, 5);

    // 3. RS-Momentum (ROC of smoothed RS) - 5 day ROC
    let rs_mom = rs_ratio.map((v, i, arr) => {
        if (i < 5 || v === null || arr[i-5] === null) return null;
        return (v / arr[i-5] - 1) * 200; // Increased base scaling
    });

    // 4. Final Smoothed Momentum - SMA 3
    let rs_mom_smooth = sms(rs_mom.map(v => v === null ? 0 : v), 3);

    // Build the trail for only the last 10 points (valid points)
    let trail = [];
    for (let i = rs_ratio.length - 10; i < rs_ratio.length; i++) {
        if (rs_ratio[i] === null) continue;
        trail.push({
            x: rs_ratio[i],
            y: rs_mom_smooth[i] || 0
        });
    }

    return trail;
}

// TEST RUN
const results = Object.keys(stockHist).slice(0, 50).map(t => ({ ticker: t, coords: getRRGCoords(t, stockHist[t] || []) })).filter(r => r.coords.length > 5);

// Analyze Spread
const allX = results.flatMap(r => r.coords.map(c => c.x));
const allY = results.flatMap(r => r.coords.map(c => c.y));

console.log("X Spread:", Math.min(...allX).toFixed(2), "to", Math.max(...allX).toFixed(2), "Std:", std(allX).toFixed(2));
console.log("Y Spread:", Math.min(...allY).toFixed(2), "to", Math.max(...allY).toFixed(2), "Std:", std(allY).toFixed(2));

// Generate a mock SVG to "convince" ourself
let svg = `<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg" style="background:#000">`;
const CW=800, CH=600, P=50;

// Viewport Scaling Logic (DRILL DOWN 1)
const medX = median(allX);
const medY = median(allY);
const stdX = std(allX) || 1;
const stdY = std(allY) || 1;

// Use 2 Standard Deviations for the viewport (auto-fills the space)
const maxRangeX = stdX * 2.5;
const maxRangeY = stdY * 2.5;

const tX = x => P + ((x - (medX - maxRangeX)) / (maxRangeX * 2)) * (CW - P * 2);
const tY = y => (CH - P) - ((y - (medY - maxRangeY)) / (maxRangeY * 2)) * (CH - P * 2);

results.forEach(r => {
    let path = `M ${tX(r.coords[0].x)} ${tY(r.coords[0].y)}`;
    for(let i=1; i<r.coords.length; i++) {
        path += ` L ${tX(r.coords[i].x)} ${tY(r.coords[i].y)}`;
    }
    svg += `<path d="${path}" stroke="cyan" fill="none" opacity="0.6" stroke-width="2"/>`;
    const last = r.coords[r.coords.length-1];
    svg += `<text x="${tX(last.x)+5}" y="${tY(last.y)}" fill="white" font-size="10">${r.ticker}</text>`;
});

svg += `</svg>`;
fs.writeFileSync('scratch/lab_test_final.svg', svg);
console.log("SVG Mock save to scratch/lab_test_final.svg");
