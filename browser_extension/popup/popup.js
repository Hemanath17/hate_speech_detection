/**
 * Popup Script
 * Handles popup UI and settings
 */

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const enableToggle = document.getElementById('enableToggle');
    const statusEl = document.getElementById('status');
    const apiStatusEl = document.getElementById('apiStatus');
    const detectionCountEl = document.getElementById('detectionCount');
    const testBtn = document.getElementById('testBtn');
    const testSection = document.getElementById('testSection');
    const testInput = document.getElementById('testInput');
    const testSubmit = document.getElementById('testSubmit');
    const testResult = document.getElementById('testResult');
    const settingsBtn = document.getElementById('settingsBtn');

    // Load settings
    chrome.storage.sync.get(['extensionEnabled', 'apiUrl'], (result) => {
        enableToggle.checked = result.extensionEnabled !== false;
        updateStatus();
    });

    // Check API health
    checkAPIHealth();

    // Toggle enable/disable
    enableToggle.addEventListener('change', (e) => {
        chrome.storage.sync.set({ extensionEnabled: e.target.checked }, () => {
            updateStatus();
            // Notify content script
            chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
                chrome.tabs.sendMessage(tabs[0].id, {
                    action: 'toggle',
                    enabled: e.target.checked
                });
            });
        });
    });

    // Test button
    testBtn.addEventListener('click', () => {
        testSection.classList.toggle('hidden');
        if (!testSection.classList.contains('hidden')) {
            testInput.focus();
        }
    });

    // Test submission
    testSubmit.addEventListener('click', async () => {
        const text = testInput.value.trim();
        if (!text) {
            testResult.innerHTML = '<p class="error">Please enter text to test</p>';
            return;
        }

        testResult.innerHTML = '<p>Testing...</p>';

        try {
            const response = await fetch('http://localhost:5000/api/detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            const result = await response.json();
            displayTestResult(result);
        } catch (error) {
            testResult.innerHTML = `<p class="error">Error: ${error.message}<br>Make sure the API server is running.</p>`;
        }
    });

    // Settings button
    settingsBtn.addEventListener('click', () => {
        chrome.runtime.openOptionsPage();
    });

    function updateStatus() {
        statusEl.textContent = enableToggle.checked ? 'Active' : 'Disabled';
        statusEl.className = enableToggle.checked ? 'status-value active' : 'status-value disabled';
    }

    async function checkAPIHealth() {
        try {
            const response = await fetch('http://localhost:5000/health', {
                method: 'GET',
                mode: 'cors',
                headers: {
                    'Accept': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            apiStatusEl.textContent = data.status === 'healthy' ? 'Connected' : 'Error';
            apiStatusEl.className = data.status === 'healthy' ? 'status-value active' : 'status-value error';
            apiStatusEl.title = data.status === 'healthy' ? 'API server is running' : 'API server returned an error';
        } catch (error) {
            apiStatusEl.textContent = 'Not Running';
            apiStatusEl.className = 'status-value error';
            apiStatusEl.title = 'API server is not running. Start it with: python src/api/app.py or ./start_api.sh';
            
            // Show helpful message in popup
            const errorMsg = document.createElement('div');
            errorMsg.className = 'api-error-message';
            errorMsg.innerHTML = `
                <p style="color: #d32f2f; font-size: 12px; margin-top: 10px; padding: 8px; background: #ffebee; border-radius: 4px;">
                    <strong>API Server Not Running</strong><br>
                    To start the server, run in terminal:<br>
                    <code style="background: #fff; padding: 2px 4px; border-radius: 2px;">python src/api/app.py</code>
                </p>
            `;
            
            // Remove existing error message if any
            const existing = document.querySelector('.api-error-message');
            if (existing) existing.remove();
            
            // Insert after API status
            apiStatusEl.parentElement.appendChild(errorMsg);
        }
    }

    function displayTestResult(result) {
        const isHate = result.is_hate;
        const confidence = (result.confidence * 100).toFixed(1);
        const hateProb = (result.hate_probability * 100).toFixed(1);
        const language = result.language.toUpperCase();

        testResult.innerHTML = `
            <div class="test-result-content ${isHate ? 'hate' : 'neutral'}">
                <h4>${isHate ? '⚠️ Hate Speech Detected' : '✓ Not Hate Speech'}</h4>
                <p><strong>Language:</strong> ${language}</p>
                <p><strong>Confidence:</strong> ${confidence}%</p>
                <p><strong>Hate Probability:</strong> ${hateProb}%</p>
                ${result.keyword_override ? '<p class="info">⚠️ Keyword override applied</p>' : ''}
            </div>
        `;
    }

    // Load detection count
    chrome.storage.local.get(['detectionCount'], (result) => {
        detectionCountEl.textContent = result.detectionCount || 0;
    });
});

