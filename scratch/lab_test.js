const fs = require('fs');

const webData = JSON.parse(fs.readFileSync('output/web_data.json', 'utf8'));
const stockHist = JSON.parse(fs.readFileSync('output/stock_history.json', 'utf8'));

const allStocks = (webData.all_stocks || []).concat(webData.stocks || []).concat(webData.pulse || []);
const unq = {};
allStocks.forEach(s => { if (!unq[s.ticker] || s.rs > (unq[s.ticker].rs || 0)) unq[s.ticker] = s; });
const stocks = Object.keys(unq).map(k => unq[k]);

console.log('Total unique stocks:', stocks.length);

const filtered = stocks.filter(s => (s.rs || 0) >= 90).sort((a,b) => b.rs - a.rs).slice(0, 40);
console.log('Filtered (Top 40 RS >= 90):', filtered.length);

    const h = (stockHist[s.ticker] || []).slice(0, 17);
    const xRaw = +(s.rs_raw || 0);
    const todayRank = +(s.rank || 0);
    const yesterdayRank = +(h[1] || todayRank);
    const weekAgoRank = +(h[7] || yesterdayRank);
    
    // Momentum = Vertical change in absolute rank (inverted because 1 is top)
    const yComposite = (weekAgoRank - todayRank) * 2; 

    const rawTrail = [];
    for (let t = 0; t < Math.min(10, h.length - 7); t++) {
        const trRank = +(h[t] || todayRank);
        const trXRaw = xRaw * ((101 - (trRank/14.47)) / (s.rs || 1)); // Approx RS ratio from rank
        const trPrevRank = +(h[t+1] || trRank);
        const trailYMom = (trPrevRank - trRank) * 5;
        rawTrail.push({ x: trXRaw, y: trailYMom });
    }
    return { ticker: s.ticker, xRaw, yComposite, trail: rawTrail.reverse() };
});

const medX = 0; // Absolute Centering as per last fix
const medY = 0;

const processed = rawData.map(d => ({
    ticker: d.ticker,
    xVal: d.xRaw - medX,
    yVal: d.yComposite - medY,
    trail: d.trail.map(tp => ({ x: tp.x - medX, y: tp.y - medY }))
}));

const allX = [], allY = [];
processed.forEach(s => {
    allX.push(s.xVal); allY.push(s.yVal);
    s.trail.forEach(tp => { allX.push(tp.x); allY.push(tp.y); });
});

let maxAbsX = Math.max(...allX.map(Math.abs)) || 1;
let maxAbsY = Math.max(...allY.map(Math.abs)) || 1;
if (maxAbsX < 0.5) maxAbsX = 0.5;
if (maxAbsY < 5.0) maxAbsY = 5.0;

console.log('Viewport Bounds:', { maxAbsX, maxAbsY });

const P = 40, CW = 500, CH = 500;
const tX = v => P + ((v + maxAbsX) / (maxAbsX * 2)) * (CW - P * 2);
const tY = v => (CH - P) - ((v + maxAbsY) / (maxAbsY * 2)) * (CH - P * 2);

processed.slice(0, 5).forEach(s => {
    console.log(`Stock ${s.ticker}: xRaw=${s.xVal.toFixed(2)}, yComp=${s.yVal.toFixed(2)} -> Screen: (${tX(s.xVal).toFixed(1)}, ${tY(s.yVal).toFixed(1)})`);
});
