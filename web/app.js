// Initialize
document.addEventListener("DOMContentLoaded", () => {
    document.getElementById('year').textContent = new Date().getFullYear();
    checkHealth();
    loadInsights();
    
    // Theme setup
    const toggle = document.getElementById('theme-toggle');
    const stored = localStorage.getItem('sm-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', stored);
    toggle.textContent = stored === 'dark' ? '🌞' : '🌙';
    
    toggle.addEventListener('click', () => {
        const cur = document.documentElement.getAttribute('data-theme');
        const nxt = cur === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', nxt);
        localStorage.setItem('sm-theme', nxt);
        toggle.textContent = nxt === 'dark' ? '🌞' : '🌙';
    });
});

let batchResults = [];

function switchTab(tab) {
    document.getElementById('tab-single').classList.remove('active');
    document.getElementById('tab-batch').classList.remove('active');
    document.getElementById(`tab-${tab}`).classList.add('active');
    
    if (tab === 'single') {
        document.getElementById('view-single').classList.remove('hidden');
        document.getElementById('view-batch').classList.add('hidden');
    } else {
        document.getElementById('view-batch').classList.remove('hidden');
        document.getElementById('view-single').classList.add('hidden');
    }
}

async function checkHealth() {
    const statusDot = document.querySelector('.status-dot');
    const statusLbl = document.querySelector('.status-label');
    try {
        const res = await fetch('/health');
        if (res.ok) {
            statusDot.style.background = 'var(--success)';
            statusDot.style.boxShadow = '0 0 8px var(--success)';
            statusLbl.textContent = 'API Online';
            updateMonitoringStats();
        } else {
            throw new Error();
        }
    } catch {
        statusDot.style.background = 'var(--danger)';
        statusDot.style.boxShadow = '0 0 8px var(--danger)';
        statusLbl.textContent = 'API Offline';
    }
}

async function updateMonitoringStats() {
    try {
        const res = await fetch('/metrics');
        const data = await res.json();
        document.getElementById('mon-requests').textContent = data.total_requests;
        document.getElementById('mon-latency').textContent = data.avg_latency_ms;
        document.getElementById('mon-status').textContent = 'Live';
    } catch (e) {}
}

function getSinglePayload() {
    return {
        product_price: parseFloat(document.getElementById('inp-price').value),
        quantity: parseInt(document.getElementById('inp-qty').value),
        discount_percentage: parseFloat(document.getElementById('inp-discount').value),
        distance_km: parseFloat(document.getElementById('inp-distance').value),
        expected_delivery_days: parseFloat(document.getElementById('inp-expected-delay').value),
        past_return_rate: parseFloat(document.getElementById('inp-return-rate').value),
        defect_rate: parseFloat(document.getElementById('inp-defect').value),
        product_category: document.getElementById('inp-category').value,
        shipping_mode: document.getElementById('inp-shipping').value,
        is_remote_area: parseInt(document.getElementById('inp-remote').value),
        warehouse_city: document.getElementById('inp-warehouse').value,
        customer_city: document.getElementById('inp-customer-city').value,
        base_return_tendency: parseFloat(document.getElementById('inp-tendency').value)
    };
}

async function runPrediction() {
    const btn = document.getElementById('btn-predict');
    btn.innerHTML = '⚡ Inferencing...';
    btn.disabled = true;
    
    document.getElementById('result-placeholder').classList.add('hidden');
    document.getElementById('results-live').classList.remove('hidden');
    
    // Add skeletons
    const rings = document.querySelectorAll('.gauge-fill');
    rings.forEach(r => r.style.strokeDashoffset = 326.7); // reset
    document.getElementById('val-return').innerHTML = '<span class="loading-skeleton" style="width:40px;height:24px;"></span>';
    document.getElementById('val-delay').innerHTML = '<span class="loading-skeleton" style="width:40px;height:24px;"></span>';
    document.getElementById('val-csat').innerHTML = '<span class="loading-skeleton" style="width:40px;height:24px;"></span>';
    document.getElementById('val-revenue').innerHTML = '<span class="loading-skeleton" style="width:60px;height:24px;"></span>';
    document.getElementById('prescription-text').innerHTML = '<div class="loading-skeleton" style="width:80%;height:20px;margin:auto;"></div>';
    
    try {
        const payload = getSinglePayload();
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        // Return 
        document.getElementById('val-return').textContent = `${data.return_probability_pct}%`;
        setGauge('gauge-return-fill', data.return_probability_pct / 100);
        
        // Delay
        document.getElementById('val-delay').textContent = data.delay_days_estimated;
        setGauge('gauge-delay-fill', Math.min(data.delay_days_estimated / 10, 1.0));
        
        // CSAT
        document.getElementById('val-csat').textContent = data.csat_projected;
        setGauge('gauge-csat-fill', data.csat_projected / 5.0);
        
        // Revenue
        document.getElementById('val-revenue').textContent = `$${data.revenue_projected}`;
        setGauge('gauge-rev-fill', Math.min(data.revenue_projected / (payload.product_price * payload.quantity), 1.0));
        
        document.getElementById('prescription-text').textContent = data.master_prescription;
        
    } catch(err) {
        document.getElementById('prescription-text').textContent = "⚠️ Error communicating with the API. Please ensure backend is running.";
    }
    
    btn.innerHTML = '⚡ Run Inference';
    btn.disabled = false;
}

function setGauge(id, perc) {
    const el = document.getElementById(id);
    if (!el) return;
    const len = 326.7; // 2 * pi * 52
    el.style.strokeDashoffset = len * (1 - perc);
}

async function runBatchPrediction() {
    const input = document.getElementById('inp-batch-json').value;
    let payload;
    try {
        payload = JSON.parse(input);
        if(!Array.isArray(payload)) throw Error();
    } catch(e) {
        alert('Invalid JSON array provided in Batch input.');
        return;
    }
    
    const tbody = document.getElementById('batch-tbody');
    tbody.innerHTML = '<tr><td colspan="7" style="text-align: center;"><div class="spinner" style="width:24px;height:24px;margin:auto;"></div></td></tr>';
    
    try {
        const res = await fetch('/predict/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        if (data.error || !data.results) {
            alert("API Error: " + (data.detail || data.error));
            tbody.innerHTML = '<tr><td colspan="7">Failed to process batch.</td></tr>';
            return;
        }
        
        batchResults = data.results;
        document.getElementById('btn-export').disabled = false;
        
        tbody.innerHTML = '';
        data.results.forEach((r, i) => {
            const tr = document.createElement('tr');
            let badgeClass = 'ok';
            if (r.risk_badge === '🟡 Warning') badgeClass = 'warning';
            if (r.risk_badge === '🔴 Critical') badgeClass = 'critical';
            
            tr.innerHTML = `
                <td>#${String(i+1).padStart(4,'0')}</td>
                <td><span class="badge ${badgeClass}">${r.risk_badge}</span></td>
                <td>${r.return_probability_pct}%</td>
                <td>${r.delay_days_estimated}</td>
                <td>${r.csat_projected}</td>
                <td>$${r.revenue_projected}</td>
                <td style="font-size:0.8rem">${r.master_prescription}</td>
            `;
            tbody.appendChild(tr);
        });
        
    } catch(err) {
        alert("Network error.");
    }
}

function exportBatchCSV() {
    if (!batchResults.length) return;
    const headers = ["Risk Badge", "Return %", "Delay (days)", "CSAT", "Revenue ($)", "Prescription"];
    const rows = batchResults.map(r => `"${r.risk_badge}","${r.return_probability_pct}","${r.delay_days_estimated}","${r.csat_projected}","${r.revenue_projected}","${r.master_prescription}"`);
    const csvContent = "data:text/csv;charset=utf-8," + headers.join(",") + "\\n" + rows.join("\\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "supplymind_batch_triage.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// ---------------------------------------------
// SHAP INSIGHTS & CHARTS
// ---------------------------------------------
async function loadInsights() {
    try {
        const res = await fetch('/insights');
        const data = await res.json();
        
        const ins = data.insights;
        if (ins.return_predictor) renderChart('chart-return', ins.return_predictor.top_features, 'var(--danger)');
        if (ins.delay_forecaster) renderChart('chart-delay', ins.delay_forecaster.top_features, 'var(--warning)');
        if (ins.csat_predictor) renderChart('chart-csat', ins.csat_predictor.top_features, 'var(--purple)');
        if (ins.revenue_predictor) renderChart('chart-revenue', ins.revenue_predictor.top_features, 'var(--primary)');
        
        // Build table
        const tb = document.getElementById('metrics-tbody');
        tb.innerHTML = '';
        
        const models = [
            {n: 'Return Predictor', t: 'XGBClassifier', m: 'ROC-AUC', k: 'roc_auc', i: ins.return_predictor},
            {n: 'Delay Forecaster', t: 'XGBRegressor', m: 'R² Score', k: 'r2_score', i: ins.delay_forecaster},
            {n: 'CSAT Predictor', t: 'XGBRegressor', m: 'R² Score', k: 'r2_score', i: ins.csat_predictor},
            {n: 'Revenue Estimator', t: 'XGBRegressor', m: 'R² Score', k: 'r2_score', i: ins.revenue_predictor}
        ];
        
        models.forEach(x => {
            const tr = document.createElement('tr');
            const val = (x.i && x.i.metrics && x.i.metrics[x.k]) ? Number(x.i.metrics[x.k]).toFixed(3) : '---';
            let status = val !== '---' ? '<span class="status-dot"></span> Active' : 'Offline';
            if (val !== '---' && Number(val) < 0) status = 'Needs Tuning';
            
            tr.innerHTML = `<td>${x.n}</td><td><span class="feature-type">${x.t}</span></td><td>${x.m}</td><td><strong>${val}</strong></td><td>${status}</td>`;
            tb.appendChild(tr);
        });
        
    } catch(err) {
        console.error('Insights missing', err);
    }
}

function renderChart(id, features, color) {
    if (!features || !features.length) return;
    const ctx = document.getElementById(id);
    if (!ctx) return;
    
    const labels = features.map(f => f.feature);
    const data = features.map(f => f.importance);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: color,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false, grid: { display: false } },
                y: { grid: { display: false }, ticks: { color: 'var(--text-color-light)', font: {size: 10} } }
            }
        }
    });
}
