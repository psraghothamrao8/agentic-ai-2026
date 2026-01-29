const API_BASE = "/api";

// Global state
let selectedIssues = new Set();
let availableClasses = [];
let allIssues = [];
let isTraining = false;

/**
 * Initial Stats & Environment Setup
 */
async function fetchStats(shouldRedirect = true) {
    try {
        const res = await fetch(`${API_BASE}/status`);
        const data = await res.json();
        renderStats(data.dataset_stats);

        // Populate available classes for dropdowns
        const clsRes = await fetch(`${API_BASE}/get_classes`);
        const clsData = await clsRes.json();
        availableClasses = clsData.classes || [];
        updateBatchDropdown();

        // Sync Phase 4 (AutoML) training state if already running
        if (data.auto_training_state && data.auto_training_state.status !== "idle") {
            const status = data.auto_training_state.status;

            // Should force show the training section if we have any state
            // BUT ONLY if allowed to redirect
            if (shouldRedirect && ["exploring", "diagnosing", "completed", "failed", "waiting_user"].includes(status)) {
                forceShowTraining();
            }

            // Only start polling if currently running
            if (["exploring", "diagnosing"].includes(status)) {
                if (!isTraining) {
                    startPollingStatus();
                }
            } else {
                // For static states (completed, failed, waiting_user), just update UI once
                updateAutoTrainingUI(data.auto_training_state);
            }
        }
    } catch (e) {
        console.error("Dashboard out of sync:", e);
        document.getElementById('system-status').innerText = "System Offline";
    }
}

function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function renderStats(stats) {
    const container = document.getElementById('stats-container');
    if (!container) return;

    let html = '';
    for (const [split, info] of Object.entries(stats)) {
        html += `
            <div class="stat-item">
                <span style="text-transform: capitalize;">${escapeHtml(split)} Set</span>
                <span>${escapeHtml(info.count)} samples</span>
            </div>
        `;
    }
    container.innerHTML = html || '<p>No data found.</p>';
}

function updateBatchDropdown() {
    const select = document.getElementById('batch-label-select');
    if (!select) return;

    // Save current value
    const curVal = select.value;
    select.innerHTML = '<option value="">Move to...</option>' +
        availableClasses.map(cls => `<option value="${escapeHtml(cls)}">${escapeHtml(cls)}</option>`).join('');
    select.value = curVal;
}

/**
 * Intelligent Agent Integration
 */
async function triggerAnalysis() {
    const output = document.getElementById('agent-output');
    const btn = document.getElementById('analyze-btn');

    if (isTraining) return alert("Cannot run analysis while training is active.");

    output.classList.add('pulse');
    output.innerHTML = "<b>Agent is analyzing dataset gradients and label consistency...</b>";
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/analyze`);
        if (!res.ok) {
            const errBody = await res.text();
            throw new Error(`Server Error (${res.status}): ${errBody.slice(0, 100)}`);
        }
        const decision = await res.json();
        displayAgentDecision(decision);
    } catch (e) {
        output.innerHTML = `<span style="color: var(--danger-color)">Analysis Error: ${e.message}</span>`;
    } finally {
        output.classList.remove('pulse');
        btn.disabled = false;
    }
}

async function evaluateCurrentModel() {
    const output = document.getElementById('agent-output');
    const btn = document.getElementById('evaluate-btn');

    if (isTraining) return alert("Cannot evaluate while training is active.");

    output.classList.add('pulse');
    output.innerHTML = "<b>Evaluating current best model...</b>";
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/evaluate_current_model`);
        if (!res.ok) {
            const errBody = await res.text();
            throw new Error(`Server Error (${res.status}): ${errBody.slice(0, 100)}`);
        }
        const results = await res.json();
        displayEvaluationResults(results);
    } catch (e) {
        output.innerHTML = `<span style="color: var(--danger-color)">Evaluation Error: ${e.message}</span>`;
    } finally {
        output.classList.remove('pulse');
        btn.disabled = false;
    }
}

function displayEvaluationResults(results) {
    const container = document.getElementById('agent-output');

    const valMetrics = results.val.metrics;
    const testMetrics = results.test ? results.test.metrics : null;

    container.innerHTML = `
        <div style="margin-bottom: 12px;">
            <b style="color: var(--accent-color)">MODEL EVALUATION RESULTS</b>
        </div>
        <div style="display: grid; grid-template-columns: ${testMetrics ? '1fr 1fr' : '1fr'}; gap: 20px; margin-top: 15px;">
            <div style="padding: 15px; background: rgba(56, 189, 248, 0.1); border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.3);">
                <h3 style="color: var(--accent-color); margin-bottom: 15px; font-size: 1rem;">Validation Set</h3>
                <div style="display: grid; gap: 10px; font-size: 0.9rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Accuracy:</span>
                        <b style="color: var(--success-color)">${(valMetrics.accuracy * 100).toFixed(2)}%</b>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Balanced Acc:</span>
                        <b>${(valMetrics.balanced_acc * 100).toFixed(2)}%</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
                        <span>Miss Rate:</span>
                        <b style="color: var(--danger-color)">${(valMetrics.miss_rate * 100).toFixed(2)}%</b>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Overkill Rate:</span>
                        <b style="color: var(--warning-color)">${(valMetrics.overkill_rate * 100).toFixed(2)}%</b>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Macro F1:</span>
                        <b>${(valMetrics.macro_f1 * 100).toFixed(2)}%</b>
                    </div>
                </div>
            </div>
            ${testMetrics ? `
            <div style="padding: 15px; background: rgba(34, 197, 94, 0.1); border-radius: 12px; border: 1px solid rgba(34, 197, 94, 0.3);">
                <h3 style="color: var(--success-color); margin-bottom: 15px; font-size: 1rem;">Test Set</h3>
                <div style="display: grid; gap: 10px; font-size: 0.9rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Accuracy:</span>
                        <b style="color: var(--success-color)">${(testMetrics.accuracy * 100).toFixed(2)}%</b>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Balanced Acc:</span>
                        <b>${(testMetrics.balanced_acc * 100).toFixed(2)}%</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
                        <span>Miss Rate:</span>
                        <b style="color: var(--danger-color)">${(testMetrics.miss_rate * 100).toFixed(2)}%</b>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Overkill Rate:</span>
                        <b style="color: var(--warning-color)">${(testMetrics.overkill_rate * 100).toFixed(2)}%</b>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Macro F1:</span>
                        <b>${(testMetrics.macro_f1 * 100).toFixed(2)}%</b>
                    </div>
                </div>
            </div>
            ` : ''}
        </div>
        <div style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px; font-size: 0.85rem; color: var(--text-secondary);">
            <b>Model:</b> ${results.model_path ? results.model_path.split(/[\\/]/).pop() : 'best_model.pth'}
        </div>
    `;
}

function displayAgentDecision(decision) {
    const container = document.getElementById('agent-output');
    const cleaningSec = document.getElementById('cleaning-section');
    const trainingSec = document.getElementById('training-section');

    container.innerHTML = `
        <div style="margin-bottom: 12px;">
            <b style="color: var(--accent-color)">AGENT INSIGHT:</b> ${decision.analysis || decision.decision}
        </div>
        <div style="padding: 10px; background: rgba(56, 189, 248, 0.1); border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <b style="color: var(--success-color)">RECOMMENDATION:</b> ${escapeHtml(decision.recommended_action.replace('_', ' '))}
            </div>
            ${decision.recommended_action === "data_cleaning" ? `<button class="btn-success" onclick="skipToBenchmark()" style="padding: 4px 10px; height: auto; min-height: unset; font-size: 0.75rem;">Skip & Continue</button>` : ''}
        </div>
    `;

    // Display Cleaning Section if "data_cleaning" recommended OR if issues exist
    if (decision.recommended_action === "data_cleaning" || (decision.issues_list && decision.issues_list.length > 0)) {
        cleaningSec.style.display = 'block';
        cleaningSec.scrollIntoView({ behavior: 'smooth' });
        renderIssues(decision.issues_list);

        // If the action was "start_training" but we are forcing cleaning view due to issues,
        // we might want to also ensure the training section is available or hidden.
        // For now, standard behavior is cleaning blocks training until "Proceed" is clicked.
        trainingSec.style.display = 'none';
    } else {
        trainingSec.style.display = 'block';
        trainingSec.scrollIntoView({ behavior: 'smooth' });
        cleaningSec.style.display = 'none';
    }
}

/**
 * Data Cleaning & Batch Logic
 */
function renderIssues(issues) {
    allIssues = issues;
    const container = document.getElementById('issues-container');
    const countPill = document.getElementById('issue-count-pill');

    countPill.innerText = issues.length;

    if (issues.length === 0) {
        container.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 40px; border: 2px dashed var(--card-border); border-radius: 20px;">
                <h3 style="color: var(--success-color)">✨ Dataset is Clean!</h3>
                <p style="color: var(--text-secondary)">No further issues detected by the agent.</p>
                <p style="color: var(--accent-color); margin-top: 20px; font-size: 0.9rem;">
                    When ready, click <b>"Proceed to Benchmark →"</b> in the toolbar above to continue.
                </p>
            </div>
        `;
        return;
    }

    container.innerHTML = issues.map((issue, idx) => `
        <div class="issue-card" id="card-${idx}">
            <div style="position: relative;">
                <input type="checkbox" class="issue-checkbox" 
                       onchange="updateSelection(${idx}, this.checked)" 
                       ${selectedIssues.has(issue.file_path) ? 'checked' : ''}
                       style="position: absolute; top: 15px; left: 15px; z-index: 10;">
                <img src="/dataset/${escapeHtml(issue.split)}/${escapeHtml(issue.given_label)}/${escapeHtml(fileName(issue.file_path))}?t=${Date.now()}" 
                     class="issue-img" loading="lazy">
            </div>
            <div class="issue-details">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px; align-items: center;">
                    <span class="badge ${escapeHtml(issue.issue_type)}">${escapeHtml(issue.issue_type.replace('_', ' ').toUpperCase())}</span>
                    <span style="font-size: 0.75rem; color: var(--accent-color)">${(issue.confidence * 100).toFixed(0)}% Conf.</span>
                </div>
                <div style="margin-bottom: 15px;">
                    <p style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 4px;">Current: <span style="color: var(--danger-color); font-weight: 600;">${escapeHtml(issue.given_label)}</span></p>
                    <p style="font-size: 0.8rem; color: var(--text-secondary);">Suggest: <span style="color: ${issue.suggested_label === 'delete' ? 'var(--danger-color)' : 'var(--success-color)'}; font-weight: 600;">${escapeHtml(issue.suggested_label.toUpperCase())}</span></p>
                </div>
                <select id="select-${idx}" style="margin-bottom: 12px;">
                    ${availableClasses.map(cls => `<option value="${escapeHtml(cls)}" ${cls === issue.suggested_label ? 'selected' : ''}>${escapeHtml(cls)}</option>`).join('')}
                </select>
                <div class="actions">
                    <button class="btn-success" onclick="applyFixSingle(${idx}, 'move')" style="padding: 6px;">Move</button>
                    <button class="btn-danger" onclick="applyFixSingle(${idx}, 'delete')" style="padding: 6px;">Delete</button>
                </div>
            </div>
        </div>
    `).join('');
}

function fileName(path) {
    return path.split(/[\\/]/).pop();
}

function updateSelection(idx, isChecked) {
    const path = allIssues[idx].file_path;
    if (isChecked) selectedIssues.add(path);
    else selectedIssues.delete(path);
}

function toggleSelectAll() {
    const masterCb = document.getElementById('select-all-issues');
    const cbs = document.querySelectorAll('.issue-checkbox');

    selectedIssues.clear();
    cbs.forEach((cb, idx) => {
        cb.checked = masterCb.checked;
        if (masterCb.checked) selectedIssues.add(allIssues[idx].file_path);
    });
}

async function applyBatchFix() {
    const targetLabel = document.getElementById('batch-label-select').value;
    if (selectedIssues.size === 0) return alert("Select items first!");
    if (!targetLabel) return alert("Select a target move label!");

    try {
        const res = await fetch(`${API_BASE}/batch_fix`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_paths: Array.from(selectedIssues),
                action: 'move',
                new_label: targetLabel
            })
        });

        if (res.ok) {
            allIssues = allIssues.filter(i => !selectedIssues.has(i.file_path));
            selectedIssues.clear();
            renderIssues(allIssues);
            fetchStats(false);
        }
    } catch (e) {
        alert("Batch fix failed: " + e.message);
    }
}

async function downloadCSV() {
    if (allIssues.length === 0) return alert("No issues to download!");

    try {
        const res = await fetch(`${API_BASE}/download_issues_csv_file`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ issues: allIssues })
        });

        if (res.ok) {
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = "detected_issues.csv";
            document.body.appendChild(a);
            a.click();
            a.remove();
        } else {
            alert("Download failed");
        }
    } catch (e) {
        alert("Error downloading CSV: " + e.message);
    }
}

async function autoFixAll() {
    if (allIssues.length === 0) return;
    if (!confirm(`Apply all ${allIssues.length} logical suggestions?`)) return;

    try {
        const items = allIssues.map(i => ({ file_path: i.file_path, new_label: i.suggested_label }));
        const res = await fetch(`${API_BASE}/batch_fix_suggestions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ items })
        });

        if (res.ok) {
            allIssues = [];
            selectedIssues.clear();
            renderIssues([]);
            fetchStats(false);
        }
    } catch (e) {
        alert("Auto-fix failed");
    }
}

async function applyFixSingle(idx, action) {
    const issue = allIssues[idx];
    const newLabel = document.getElementById(`select-${idx}`).value;

    try {
        const res = await fetch(`${API_BASE}/fix_issue`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_path: issue.file_path, action, new_label: newLabel })
        });

        if (res.ok) {
            allIssues.splice(idx, 1);
            selectedIssues.delete(issue.file_path);
            renderIssues(allIssues);
            fetchStats(false);
        }
    } catch (e) {
        alert("Action failed");
    }
}

function skipToBenchmark() {
    const cleaningSec = document.getElementById('cleaning-section');
    const trainingSec = document.getElementById('training-section');
    cleaningSec.style.display = 'none';
    trainingSec.style.display = 'block';
    trainingSec.scrollIntoView({ behavior: 'smooth' });
}

function forceShowTraining() {
    skipToBenchmark();
}


/**
 * AutoML Benchmarking Logic
 */
async function startTraining() {
    if (isTraining) return;

    try {
        const res = await fetch(`${API_BASE}/start_auto_training`, { method: 'POST' });

        if (!res.ok) {
            const errorData = await res.json().catch(() => ({ detail: 'Unknown error' }));

            // Check if the error is about training already in progress
            if (res.status === 400 && errorData.detail && errorData.detail.includes('already in progress')) {
                // Offer to reset the stuck state
                const shouldReset = confirm(
                    `${errorData.detail}\n\n` +
                    `It seems the training state is stuck. No actual training is running.\n\n` +
                    `Would you like to reset the training state and try again?`
                );

                if (shouldReset) {
                    await resetTrainingState();
                    // Try starting again after reset
                    await startTraining();
                }
            } else {
                alert(`Failed to start training: ${errorData.detail || 'Unknown error'}`);
            }
            return;
        }

        startPollingStatus();
    } catch (e) {
        alert(`Could not initiate benchmarking: ${e.message}`);
    }
}

async function resetTrainingState() {
    try {
        const res = await fetch(`${API_BASE}/reset_training_state`, { method: 'POST' });
        const data = await res.json();

        if (res.ok) {
            console.log(`Training state reset: ${data.message}`);
            // Refresh the UI
            await fetchStats();
        } else {
            alert('Failed to reset training state');
        }
    } catch (e) {
        alert(`Error resetting state: ${e.message}`);
    }
}

async function performSoftReset() {
    if (!confirm("Soft Reset: This will clear current training state and logs from the dashboard. Your Data and Models will be preserved.\n\nProceed?")) return;

    try {
        const res = await fetch(`${API_BASE}/reset_training_state?force_delete=true`, { method: 'POST' }); // force_delete here clears metrics.json
        const data = await res.json();

        if (res.ok) {
            alert("Soft Reset Complete! Dashboard will reload.");
            window.location.reload();
        } else {
            alert('Failed to reset: ' + data.detail);
        }
    } catch (e) {
        alert(`Error: ${e.message}`);
    }
}

async function performHardReset() {
    const confirmation = prompt("⚠️ HARD RESET WARNING ⚠️\n\nThis will DELETE ALL:\n- Uploaded Datasets\n- Trained Models\n- Logs\n\nThis action cannot be undone.\n\nType 'DELETE' to confirm:");

    if (confirmation !== 'DELETE') {
        if (confirmation !== null) alert("Reset cancelled. You must type 'DELETE' exactly.");
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/reset_training_state?hard_reset=true`, { method: 'POST' });
        const data = await res.json();

        if (res.ok) {
            alert("Hard Reset Successful. System is essentially brand new.");
            window.location.reload();
        } else {
            alert('Failed to hard reset: ' + data.detail);
        }
    } catch (e) {
        alert(`Error: ${e.message}`);
    }
}

// Deprecated but kept to avoid breakages if called elsewhere
function fullSystemReset() {
    performSoftReset();
}


function startPollingStatus() {
    isTraining = true;
    const btn = document.getElementById('train-btn');
    const statusBadge = document.getElementById('system-status');
    btn.disabled = true;
    document.getElementById('analyze-btn').disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Benchmarking...`;
    statusBadge.innerText = "Auto-Benchmarking Active";
    statusBadge.classList.add('pulse');

    const interval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/auto_training_status`);
            const state = await res.json();

            updateAutoTrainingUI(state);

            // Stop polling when training is truly complete or failed
            // Keep polling during 'diagnosing' as it's a transient state
            if (["completed", "failed", "waiting_user"].includes(state.status)) {
                clearInterval(interval);
                isTraining = false;
                btn.disabled = false;
                btn.innerText = "Start Multi-Model Benchmark";
                statusBadge.innerText = "System Standby";
                statusBadge.classList.remove('pulse');
                fetchStats(false);
                document.getElementById('analyze-btn').disabled = false;
            }
        } catch (e) {
            console.error("Polling error:", e);
            clearInterval(interval);
            isTraining = false;

            // Reset UI on error so it doesn't get stuck
            const btn = document.getElementById('train-btn');
            const statusBadge = document.getElementById('system-status');

            if (btn) {
                btn.disabled = false;
                btn.innerText = "Start Multi-Model Benchmark";
            }
            if (statusBadge) {
                statusBadge.innerText = "System Standby";
                statusBadge.classList.remove('pulse');
            }
            const analyzeBtn = document.getElementById('analyze-btn');
            if (analyzeBtn) analyzeBtn.disabled = false;
        }
    }, 2000);
}

function updateAutoTrainingUI(state) {
    const logs = document.getElementById('training-logs');
    const leaderboard = document.getElementById('leaderboard-content');

    if (state.status === "exploring" || state.status === "final_training") {
        const title = state.status === "exploring" ? "🚀 AUTO-BENCHMARKING ACTIVE" : "🏋️ FINAL MODEL TRAINING";
        const subtext = state.status === "exploring"
            ? `Fine-tuning hyperparameters using Optuna (Trial ${state.iteration})...`
            : "Performing final deep fine-tuning for maximum accuracy...";

        logs.innerHTML = `
            <div style="color: var(--accent-color); font-weight: 700; margin-bottom: 10px;">${title}</div>
            <div class="progress-bar-container" style="height: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; margin-bottom: 15px; overflow: hidden;">
                <div style="width: ${(state.current_config / state.total_configs) * 100}%; height: 100%; background: var(--accent-color);"></div>
            </div>
            <p>Evaluating Architecture <b>${state.current_config + 1}</b> / ${state.total_configs}</p>
            <p>Accuracy: <b>${(state.best_acc * 100).toFixed(2)}%</b> ${state.current_val_acc ? `<small>(Current: ${(state.current_val_acc * 100).toFixed(2)}%)</small>` : ''}</p>
            
            <div style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="display: flex; justify-content: space-between; font-size: 0.7rem; margin-bottom: 5px;">
                    <span>Epoch ${state.current_epoch || 0} / ${state.total_epochs || 0}</span>
                </div>
                <div style="height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; overflow: hidden;">
                    <div style="width: ${((state.current_epoch || 0) / (state.total_epochs || 1)) * 100}%; height: 100%; background: var(--success-color);"></div>
                </div>
            </div>

            <div style="margin-top: 20px; font-size: 0.75rem; color: var(--text-secondary);">
                ${subtext}
            </div>
        `;
    } else if (state.status === "completed") {
        // Safely handle exploration_results
        const explorationResults = state.exploration_results || {};
        const best = explorationResults.best_result || null;
        const results = explorationResults.all_results || state.results || [];

        if (best) {
            logs.innerHTML = `
                <div style="color: var(--success-color); font-weight: 700;">✅ BENCHMARK COMPLETE</div>
                <h1 style="margin: 15px 0;">${(best.val_acc * 100).toFixed(1)}% <small style="font-size: 0.5em; color: var(--text-secondary)">Val Acc</small></h1>
                
                <p><b>Winner:</b> ${best.config_name || 'Best Model'}</p>
                <p style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 10px;">Train Acc: ${(best.train_acc * 100).toFixed(1)}%</p>

                <div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 0.85rem; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
                    <div>
                        <b style="color: var(--accent-color)">Validation Set</b>
                        <div style="margin-top: 5px;">
                            <div style="display:flex; justify-content:space-between;"><span>Miss:</span> <b>${(best.miss_rate * 100).toFixed(1)}%</b></div>
                            <div style="display:flex; justify-content:space-between;"><span>Overkill:</span> <b>${(best.overkill_rate * 100).toFixed(1)}%</b></div>
                        </div>
                    </div>
                    ${best.test_metrics ? `
                    <div>
                        <b style="color: var(--success-color)">Test Set</b>
                        <div style="margin-top: 5px;">
                            <div style="display:flex; justify-content:space-between;"><span>Miss:</span> <b>${(best.test_metrics.miss_rate * 100).toFixed(1)}%</b></div>
                            <div style="display:flex; justify-content:space-between;"><span>Overkill:</span> <b>${(best.test_metrics.overkill_rate * 100).toFixed(1)}%</b></div>
                        </div>
                    </div>
                    ` : '<div style="color: var(--text-secondary); font-style: italic;">No Test Set Found</div>'}
                </div>
            `;

            // Check for diagnosis and append analysis
            const diagnosis = state.diagnosis || {};
            if (diagnosis.conclusion || diagnosis.dataset_analysis) {
                logs.innerHTML += `
                    <div style="margin-top: 20px; padding: 15px; background: rgba(56, 189, 248, 0.1); border-radius: 8px; font-size: 0.9rem;">
                        <p style="color: var(--accent-color); font-weight: 600; margin-bottom: 8px;">🧠 AI Analysis:</p>
                        ${diagnosis.conclusion ? `<p style="margin-bottom:8px"><b>Conclusion:</b> ${escapeHtml(diagnosis.conclusion)}</p>` : ''}
                        ${diagnosis.dataset_analysis ? `<p style="margin-bottom:8px"><b>Dataset:</b> ${escapeHtml(diagnosis.dataset_analysis)}</p>` : ''}
                         ${diagnosis.next_steps ? `
                            <div style="margin-top: 15px; display: flex; gap: 10px; flex-wrap: wrap;">
                                ${diagnosis.next_steps.map(step =>
                    `<button class="btn-primary" style="font-size: 0.8rem; padding: 5px 10px;" onclick="handleNextStep('${escapeHtml(step.action)}')">${escapeHtml(step.label)}</button>`
                ).join('')}
                            </div>
                        ` : ''}
                    </div>
                 `;
            }

            // Update Leaderboard with history
            if (results.length > 0) {
                leaderboard.innerHTML = results.sort((a, b) => b.val_acc - a.val_acc).map((run, i) => `
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <span style="font-size: 0.8rem; color: ${i === 0 ? 'var(--warning-color)' : 'inherit'}">${i === 0 ? '👑' : i + 1}. ${run.config_name || 'Model ' + (i + 1)}</span>
                        <span style="font-weight: 600;">${(run.val_acc * 100).toFixed(1)}%</span>
                    </div>
                `).join('');
            } else {
                leaderboard.innerHTML = '<p style="color: var(--text-secondary);">No evaluation history available.</p>';
            }
        } else {
            // Fallback if no best_result found
            logs.innerHTML = `
                <div style="color: var(--success-color); font-weight: 700;">✅ TRAINING COMPLETE</div>
                <p style="margin-top: 15px; color: var(--text-secondary);">Best model saved. Results available in logs.</p>
            `;
            leaderboard.innerHTML = '<p style="color: var(--text-secondary);">Evaluation results shown here.</p>';
        }
    } else if (state.status === "diagnosing") {
        // Show diagnosis results
        const explorationResults = state.exploration_results || {};
        const best = explorationResults.best_result || null;
        const diagnosis = state.diagnosis || {};

        if (best) {
            logs.innerHTML = `
                <div style="color: var(--warning-color); font-weight: 700;">🔍 DIAGNOSIS COMPLETE</div>
                <h1 style="margin: 15px 0;">${(best.val_acc * 100).toFixed(1)}% <small style="font-size: 0.5em; color: var(--text-secondary)">Achieved</small></h1>
                <div style="margin-top: 15px; padding: 15px; background: rgba(251, 191, 36, 0.1); border-left: 3px solid var(--warning-color); border-radius: 8px;">
                    <p style="margin-bottom: 10px;"><b>Model:</b> ${best.config_name || 'Best Model'}</p>
                    <p style="margin-bottom: 10px;"><b>Train Acc:</b> ${(best.train_acc * 100).toFixed(1)}%</p>
                    <p style="margin-bottom: 10px;"><b>Miss Rate:</b> ${(best.miss_rate * 100).toFixed(1)}%</p>
                    <p><b>Overkill Rate:</b> ${(best.overkill_rate * 100).toFixed(1)}%</p>
                </div>
                ${diagnosis.recommendation ? `
                    <div style="margin-top: 20px; padding: 15px; background: rgba(56, 189, 248, 0.1); border-radius: 8px;">
                        <p style="color: var(--accent-color); font-weight: 600; margin-bottom: 8px;">💡 Agent Recommendation:</p>
                        <p style="font-size: 0.9rem;">${diagnosis.recommendation}</p>
                    </div>
                ` : ''}
            `;

            // Update leaderboard
            const results = explorationResults.all_results || [];
            if (results.length > 0) {
                leaderboard.innerHTML = results.sort((a, b) => b.val_acc - a.val_acc).map((run, i) => `
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <span style="font-size: 0.8rem; color: ${i === 0 ? 'var(--warning-color)' : 'inherit'}">${i === 0 ? '👑' : i + 1}. ${run.config_name || 'Model ' + (i + 1)}</span>
                        <span style="font-weight: 600;">${(run.val_acc * 100).toFixed(1)}%</span>
                    </div>
                `).join('');
            }
        } else {
            logs.innerHTML = `<div style="color: var(--warning-color);">🔍 Analyzing results...</div>`;
        }
    } else if (state.status === "failed") {
        logs.innerHTML = `<div style="color: var(--danger-color)">❌ Benchmarking failed: ${state.error}</div>`;
    }
}

async function uploadAndRun() {
    console.log("uploadAndRun triggered");
    const fileInput = document.getElementById('dataset-upload');
    const statusDiv = document.getElementById('upload-status');
    const btn = document.getElementById('upload-btn');

    if (!fileInput || !fileInput.files.length) {
        console.warn("No file selected or input not found");
        alert("Please select a zip file first.");
        return;
    }

    const file = fileInput.files[0];
    console.log(`Starting upload: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);

    const formData = new FormData();
    formData.append("file", file);

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Uploading...`;
    statusDiv.innerText = "Transferring data to server... (this may take a minute for large files)";
    statusDiv.style.color = "var(--accent-color)";

    try {
        console.log("Sending fetch request to /api/upload_dataset");
        const res = await fetch(`${API_BASE}/upload_dataset`, {
            method: 'POST',
            body: formData
        });

        console.log("Response received:", res.status, res.statusText);

        if (!res.ok) {
            let errText = "Upload failed";
            try {
                const errData = await res.json();
                errText = errData.detail || errText;
            } catch (p) {
                errText = await res.text() || errText;
            }
            throw new Error(errText);
        }

        const successData = await res.json();
        console.log("Upload Success:", successData);

        statusDiv.innerText = "✅ Upload & Extraction Complete! Analyzing...";
        statusDiv.style.color = "var(--success-color)";
        btn.innerHTML = "Success!";

        alert("Upload Successful!\nDataset has been updated. The agent will now analyze the new data.");

        // Reset and run
        setTimeout(() => {
            statusDiv.innerText = "";
            fileInput.value = "";
            btn.disabled = false;
            btn.innerHTML = "🚀 Upload & Run";
            fetchStats();
            // Trigger analysis automatically
            console.log("Triggering analysis...");
            triggerAnalysis();
        }, 1000);

    } catch (e) {
        console.error("Upload Error Details:", e);
        statusDiv.innerText = "❌ Error: " + e.message;
        statusDiv.style.color = "var(--danger-color)";
        btn.disabled = false;
        btn.innerHTML = "🚀 Upload & Run";
        alert("Upload Failed: " + e.message + "\nCheck browser console (F12) for more details.");
    }
}

// Global scope expose
window.uploadAndRun = uploadAndRun;
window.performSoftReset = performSoftReset;
window.performHardReset = performHardReset;
// window.toggleSelectAll = toggleSelectAll; // Ensure this is also exposed if used in HTML
// window.downloadCSV = downloadCSV; // and this
window.autoFixAll = autoFixAll;
window.applyBatchFix = applyBatchFix;
window.skipToBenchmark = skipToBenchmark;
window.startTraining = startTraining;
window.evaluateCurrentModel = evaluateCurrentModel;
window.triggerAnalysis = triggerAnalysis;
window.downloadCSV = downloadCSV;
window.toggleSelectAll = toggleSelectAll;
window.updateSelection = updateSelection;
window.applyFixSingle = applyFixSingle;
window.triggerAnalysis = triggerAnalysis;
window.evaluateCurrentModel = evaluateCurrentModel;
window.startTraining = startTraining;
window.autoFixAll = autoFixAll;
window.applyBatchFix = applyBatchFix;
window.updateSelection = updateSelection;
window.toggleSelectAll = toggleSelectAll;
window.applyFixSingle = applyFixSingle;
window.skipToBenchmark = skipToBenchmark;
window.forceShowTraining = forceShowTraining;
window.handleNextStep = handleNextStep;
window.performSoftReset = performSoftReset;
window.performHardReset = performHardReset;

// [Removed Duplicate Functions]

async function handleNextStep(action) {
    if (action === 'filter_dataset') {
        const cleaningSec = document.getElementById('cleaning-section');
        const trainingSec = document.getElementById('training-section');
        const logs = document.getElementById('training-logs');

        // Show loading state in logs
        logs.innerHTML += `<div style="margin-top:20px; color: var(--accent-color); font-style: italic;">🔄 Running Hybrid Analysis using best model...</div>`;

        try {
            const res = await fetch(`${API_BASE}/analyze_with_model`);
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Analysis failed");
            }
            const decision = await res.json();

            // Switch views
            trainingSec.style.display = 'none';
            cleaningSec.style.display = 'block';
            cleaningSec.scrollIntoView({ behavior: 'smooth' });

            // Render new issues
            displayAgentDecision(decision);

        } catch (e) {
            alert(`Hybrid Analysis Failed: ${e.message}`);
        }
    } else if (action === 'more_tuning') {
        // Reset state and restart training
        if (confirm("This will reset the current results and start a new hyperparameter tuning session. Continue?")) {
            await resetTrainingState();
            await startTraining();
        }
    }
}

// Init
document.addEventListener('DOMContentLoaded', fetchStats);
