import os

html_path = r"d:\coding_stuffs\minor_project_supply_chain\Order-Return-Rate-Dataset\web\index.html"
with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

# Add the expected_delivery_days input correctly in the single predictions
new_form_group = """
                    <div class="form-group">
                        <label for="inp-expected-delay">Expected Delay</label>
                        <input type="number" id="inp-expected-delay" value="3.0" min="1" max="30" step="0.5">
                    </div>
"""
if 'for="inp-expected-delay"' not in content:
    content = content.replace(
        '<div class="form-group">\n                        <label for="inp-distance">',
        new_form_group + '                    <div class="form-group">\n                        <label for="inp-distance">'
    )

# Add the batch prediction UI block right after the single prediction block closes
batch_ui = """
        </div>
        
        <div id="view-batch" class="batch-layout hidden glass-card">
            <h3 class="form-title">Upload Batch Orders (CSV) or Paste JSON</h3>
            <p class="batch-desc">Process up to 1000 orders at once. Get instant triage risk scores, delays, and return predictions.</p>
            <div class="batch-controls">
                <textarea id="inp-batch-json" rows="6" placeholder='[{"product_price": 120.0, "quantity": 1, ...}]'></textarea>
                <div class="batch-actions">
                    <button class="btn btn-primary" onclick="runBatchPrediction()">Process Batch</button>
                    <button class="btn btn-outline" onclick="exportBatchCSV()" id="btn-export" disabled>Export to CSV</button>
                </div>
            </div>
            
            <div class="table-wrap batch-table-wrap">
                <table id="batch-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Risk Badge</th>
                            <th>Return %</th>
                            <th>Delay (days)</th>
                            <th>CSAT</th>
                            <th>Revenue ($)</th>
                            <th>Prescription</th>
                        </tr>
                    </thead>
                    <tbody id="batch-tbody">
                        <tr><td colspan="7" style="text-align: center; color: var(--text-color-light);">No data processed yet.</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
"""

# The single view ends right before <!-- SECTION 5: INSIGHTS
# Let's target exactly `</div>\n    </div>\n</section>\n\n<!-- ═══ SECTION 5` -> `</div>\n    ` + batch_ui + `</div>\n</section>\n\n<!-- ═══ SECTION 5`
import re
if 'id="view-batch"' not in content:
    content = re.sub(
        r'(\s+)</div>\s+</div>\s+</section>\s+<!-- ═══ SECTION 5: INSIGHTS',
        r'\1</div>' + batch_ui + r'\n    </div>\n</section>\n\n<!-- ═══ SECTION 5: INSIGHTS',
        content
    )

with open(html_path, "w", encoding="utf-8") as f:
    f.write(content)
