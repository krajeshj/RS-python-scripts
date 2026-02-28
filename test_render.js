const fs = require('fs');

const data = JSON.parse(fs.readFileSync('output/web_data.json', 'utf8'));
const pool = data.all_stocks || data.stocks || [];

function createCanslimTray(canslim) {
            if (!canslim) return '';
            const letters = ['C', 'A', 'N', 'S', 'L', 'I', 'M'];
            return `<div class="canslim-tray">
                ${letters.map(l => {
                const active = canslim[l.toLowerCase()];
                return `<div class="canslim-letter ${active ? 'active' : ''}" title="${l}">${l}</div>`;
            }).join('')}
            </div>`;
}

function renderStock(s, idx) {
    const earningsBadge = (s.days_to_earnings >= 0 && s.days_to_earnings <= 14)
        ? `<div class="earnings-alert">📅 EARNINGS IN ${s.days_to_earnings}D</div>`
        : '';
    const priceStr = s.price ? `$${s.price.toFixed(2)}` : '';

    function fmtVol(v) {
        if (!v) return 'N/A';
        if (v >= 1000000) return (v / 1000000).toFixed(1) + 'M';
        if (v >= 1000) return (v / 1000).toFixed(0) + 'K';
        return v.toString();
    }

    let volStyle = "font-size: 0.65rem; color: var(--text2); padding: 2px 6px; border-radius: 4px; border: 1px solid var(--border); background: var(--surface); display: inline-block;";
    if (s.volume > s.avg_volume) {
        if (s.is_up) {
            volStyle = "font-size: 0.65rem; font-weight: 800; padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(0,230,118,0.3); background: rgba(0,230,118,0.1); color: var(--green); display: inline-block;";
        } else {
            volStyle = "font-size: 0.65rem; font-weight: 800; padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(255,145,0,0.3); background: rgba(255,145,0,0.1); color: var(--orange); display: inline-block;";
        }
    }
    const volBadge = `<div style="${volStyle}" title="Avg Vol: ${fmtVol(s.avg_volume)}">Vol: ${fmtVol(s.volume)}</div>`;

    return `
        <div class="stats-top">
            <div class="rank-badge">RANK #${s.rank}</div>
            ${s.flip ? `<div class="flip-badge">⚡ PRIME FLIP</div>` : ''}
            ${priceStr ? `<div class="source-badge">${priceStr}</div>` : ''}
            ${volBadge}
            <div class="source-badge">${s.source}</div>
        </div>
    `;
}

try {
    for (let i = 0; i < 5; i++) {
        console.log(renderStock(pool[i], i));
    }
} catch (e) {
    console.error("CRASH:", e);
}
