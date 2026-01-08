/**
 * Background Service Worker
 * Handles extension lifecycle and communication
 */

// Extension installation
chrome.runtime.onInstalled.addListener((details) => {
    if (details.reason === 'install') {
        // Set default settings
        chrome.storage.sync.set({
            extensionEnabled: true,
            detectionSensitivity: 'medium',
            apiUrl: 'http://localhost:5000/api/detect'
        });
        
        console.log('Hate Speech Detector installed');
    }
});

// Handle messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'detect') {
        // Forward to API
        fetch(request.apiUrl || 'http://localhost:5000/api/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: request.text })
        })
        .then(response => response.json())
        .then(data => sendResponse({ success: true, data: data }))
        .catch(error => sendResponse({ success: false, error: error.message }));
        
        return true; // Keep channel open for async response
    }
});

// Health check for API
async function checkAPIHealth() {
    try {
        const response = await fetch('http://localhost:5000/health');
        const data = await response.json();
        return data.status === 'healthy';
    } catch (error) {
        return false;
    }
}

// Periodic API health check
setInterval(async () => {
    const isHealthy = await checkAPIHealth();
    chrome.storage.local.set({ apiHealthy: isHealthy });
}, 30000); // Check every 30 seconds

