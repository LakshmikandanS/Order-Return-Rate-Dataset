document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    // Fetch Inputs from UI
    const price = parseFloat(document.getElementById('price').value);
    const discount = parseFloat(document.getElementById('discount').value);
    const distance = parseFloat(document.getElementById('distance').value);
    const pastReturnRate = parseFloat(document.getElementById('past_return_rate').value);
    const defectRate = parseFloat(document.getElementById('defect_rate').value);

    // API Payload
    const orderPayload = {
        product_price: price,
        quantity: 1, // Defaulting to single order for this demo
        discount_percentage: discount,
        distance_km: distance,
        past_return_rate: pastReturnRate,
        defect_rate: defectRate,
        product_category: "Electronics", // Hardcoded for frontend MVP
        shipping_mode: "Standard",
        is_remote_area: 0,
        warehouse_city: "Kolkata",
        customer_city: "Bangalore",
        base_return_tendency: 0.15
    };

    // Show Loading Stats
    document.getElementById('results-section').style.display = 'grid';
    document.getElementById('master-prescription').style.display = 'flex';
    document.getElementById('rev-est').textContent = "Model Inferencing...";
    document.getElementById('delay-est').textContent = "Model Inferencing...";
    document.getElementById('return-risk').textContent = "Model Inferencing...";
    document.getElementById('csat-est').textContent = "Model Inferencing...";
    document.getElementById('master-prescription-text').textContent = "Running ML Prescriptions...";

    try {
        // FASTAPI Integration: Call The Live XGBoost Python Models (Port 8001)
        const response = await fetch('http://127.0.0.1:8001/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(orderPayload)
        });

        const mlData = await response.json();

        // 1. REVENUE
        document.getElementById('rev-est').textContent = `$${mlData.revenue_projected}`;
        document.getElementById('rev-presc').textContent = "Via XGBRegressor (Revenue)";

        // 2. DELAY
        const delayEl = document.getElementById('delay-est');
        const delayDesc = document.getElementById('delay-presc');
        delayEl.textContent = `${mlData.delay_days_estimated} days`;
        if (mlData.delay_days_estimated > 3) {
            delayEl.className = 'card-value danger-text';
            delayDesc.innerHTML = `<span class="warning-text">Risk: Remote routing via XGBoost</span>`;
        } else {
            delayEl.className = 'card-value success-text';
            delayDesc.innerHTML = `<span class="success-text">On track</span>`;
        }

        // 3. RETURN RISK
        const returnRisk = mlData.return_probability_pct;
        const returnEl = document.getElementById('return-risk');
        const returnDesc = document.getElementById('return-presc');
        returnEl.textContent = `${returnRisk}%`;
        
        if (returnRisk > 40) {
            returnEl.className = 'card-value danger-text';
            returnDesc.innerHTML = `<span class="danger-text">High Wardrobing & Defect Risk</span>`;
        } else if (returnRisk > 20) {
            returnEl.className = 'card-value warning-text';
            returnDesc.innerHTML = `<span class="warning-text">Moderate Risk</span>`;
        } else {
            returnEl.className = 'card-value success-text';
            returnDesc.innerHTML = `<span class="success-text">Safe Order via XGBClassifier</span>`;
        }

        // 4. CSAT PREDICTOR
        const csat = mlData.csat_projected;
        const csatEl = document.getElementById('csat-est');
        const csatDesc = document.getElementById('csat-presc');
        csatEl.textContent = `${csat} / 5.0`;

        if (csat < 3.0) {
            csatEl.className = 'card-value danger-text';
            csatDesc.innerHTML = `<span class="danger-text">Poor Experience Projected</span>`;
        } else if (csat < 4.0) {
            csatEl.className = 'card-value warning-text';
            csatDesc.innerHTML = `<span class="warning-text">Average</span>`;
        } else {
            csatEl.className = 'card-value success-text';
            csatDesc.innerHTML = `<span class="success-text">Excellent</span>`;
        }

        // 5. MASTER PRESCRIPTION (Business Logic Matrix powered by the backend)
        const masterText = document.getElementById('master-prescription-text');
        masterText.innerHTML = mlData.master_prescription;

    } catch (error) {
        console.error("FastAPI Error:", error);
        document.getElementById('master-prescription-text').innerHTML = `<strong>📡 API Ofline:</strong> Could not connect to Python FastAPI backend at http://127.0.0.1:8000.`;
    }
});