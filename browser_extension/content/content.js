/**
 * Content Script for Real-Time Hate Speech Detection
 * Monitors input fields, highlights hate speech, and flags before submission
 */

(function() {
    'use strict';

    // Configuration
    const CONFIG = {
        API_URL: 'http://localhost:5000/api/detect',
        DEBOUNCE_DELAY: 500, // ms to wait after typing stops
        HIGHLIGHT_CLASS: 'hate-speech-detected',
        BORDERLINE_CLASS: 'hate-speech-borderline',
        ENABLED_KEY: 'extensionEnabled',
        SENSITIVITY_KEY: 'detectionSensitivity'
    };

    // State
    let isEnabled = true;
    let debounceTimer = null;
    let detectionCache = new Map();
    let activeHighlights = new Map();

    // Initialize
    init();

    function init() {
        // Load settings
        chrome.storage.sync.get([CONFIG.ENABLED_KEY], (result) => {
            isEnabled = result[CONFIG.ENABLED_KEY] !== false;
            if (isEnabled) {
                startMonitoring();
            }
        });

        // Listen for settings changes
        chrome.storage.onChanged.addListener((changes) => {
            if (changes[CONFIG.ENABLED_KEY]) {
                isEnabled = changes[CONFIG.ENABLED_KEY].newValue;
                if (isEnabled) {
                    startMonitoring();
                } else {
                    stopMonitoring();
                }
            }
        });

        // Re-scan comments when page becomes visible (for lazy-loaded content)
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && isEnabled) {
                setTimeout(scanExistingComments, 1000);
            }
        });

        // Scan comments when scrolling (for infinite scroll pages)
        let scrollTimeout;
        window.addEventListener('scroll', () => {
            if (!isEnabled) return;
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {
                scanExistingComments();
            }, 1000);
        });
    }

    function startMonitoring() {
        // Monitor existing input fields
        monitorInputFields();

        // Monitor dynamically added input fields
        observeNewElements();

        // Intercept form submissions
        interceptFormSubmissions();

        // Scan and highlight existing comments
        scanExistingComments();

        // Monitor for new comments
        observeNewComments();
    }

    function stopMonitoring() {
        // Remove all highlights
        removeAllHighlights();
        // Remove comment highlights
        removeAllCommentHighlights();
    }

    /**
     * Monitor all input and textarea fields
     */
    function monitorInputFields() {
        const inputs = document.querySelectorAll('input[type="text"], input[type="search"], textarea, [contenteditable="true"]');
        inputs.forEach(input => {
            if (!input.dataset.hateSpeechMonitored) {
                input.dataset.hateSpeechMonitored = 'true';
                setupInputListener(input);
            }
        });
    }

    /**
     * Setup event listeners for an input field
     */
    function setupInputListener(element) {
        // Handle typing events
        element.addEventListener('input', (e) => {
            if (!isEnabled) return;
            debounceDetection(element);
        });

        // Handle paste events
        element.addEventListener('paste', (e) => {
            if (!isEnabled) return;
            setTimeout(() => debounceDetection(element), 100);
        });

        // Handle blur (when user leaves field)
        element.addEventListener('blur', () => {
            if (!isEnabled) return;
            detectAndHighlight(element);
        });
    }

    /**
     * Debounce detection to avoid too many API calls
     */
    function debounceDetection(element) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            detectAndHighlight(element);
        }, CONFIG.DEBOUNCE_DELAY);
    }

    /**
     * Detect hate speech and highlight
     */
    async function detectAndHighlight(element) {
        const text = getTextFromElement(element);
        if (!text || text.trim().length < 3) {
            removeHighlights(element);
            return;
        }

        // Quick keyword check first (fast, local)
        if (!hasSuspiciousKeywords(text)) {
            removeHighlights(element);
            return;
        }

        // Check cache
        const cacheKey = text.toLowerCase().trim();
        if (detectionCache.has(cacheKey)) {
            const result = detectionCache.get(cacheKey);
            highlightText(element, result);
            return;
        }

        // Call API
        const result = await detectHateSpeech(text);
        
        // Only cache successful results (no error)
        if (!result.error) {
            detectionCache.set(cacheKey, result);
        }
        
        // Only highlight if we got a valid result
        if (!result.error) {
            highlightText(element, result);
        } else {
            // Show a subtle indicator that API is unavailable
            console.warn('Detection unavailable:', result.error);
        }
    }

    /**
     * Quick keyword check (local, fast)
     */
    function hasSuspiciousKeywords(text) {
        const keywords = [
            'stupid', 'idiot', 'hate', 'kill', 'die', 'worthless',
            'useless', 'trash', 'garbage', 'scum', 'retard'
        ];
        const textLower = text.toLowerCase();
        return keywords.some(keyword => textLower.includes(keyword));
    }

    /**
     * Call API to detect hate speech
     */
    async function detectHateSpeech(text) {
        try {
            // Direct fetch without health check to avoid double requests
            const response = await fetch(CONFIG.API_URL, {
                method: 'POST',
                mode: 'cors',
                credentials: 'omit',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ text: text })
            }).catch((fetchError) => {
                // Network error (server not running, CORS blocked, etc.)
                console.warn('Fetch failed:', fetchError.message);
                throw new Error(`Cannot connect to API server. Make sure it's running on ${CONFIG.API_URL}`);
            });

            if (!response.ok) {
                const errorText = await response.text().catch(() => 'Unknown error');
                throw new Error(`API error: ${response.status} ${response.statusText} - ${errorText}`);
            }

            const result = await response.json();
            
            // Validate result structure
            if (!result || typeof result !== 'object') {
                throw new Error('Invalid response from API');
            }
            
            return result;
        } catch (error) {
            // Log error but don't throw - return neutral result instead
            console.warn('Detection error:', error.message);
            
            // Return neutral result on error
            return {
                is_hate: false,
                confidence: 0.0,
                hate_probability: 0.0,
                neutral_probability: 1.0,
                language: 'eng',
                error: error.message || 'Unknown error'
            };
        }
    }

    /**
     * Highlight text based on detection result
     */
    function highlightText(element, result) {
        if (!result.is_hate && result.hate_probability < 0.3) {
            removeHighlights(element);
            return;
        }

        const text = getTextFromElement(element);
        if (!text) return;

        // Remove existing highlights
        removeHighlights(element);

        // Determine highlight class
        const highlightClass = result.is_hate 
            ? CONFIG.HIGHLIGHT_CLASS 
            : CONFIG.BORDERLINE_CLASS;

        // For contenteditable divs
        if (element.contentEditable === 'true') {
            highlightContentEditable(element, highlightClass, result);
        } 
        // For input/textarea
        else {
            highlightInputField(element, highlightClass, result);
        }

        // Store highlight info
        activeHighlights.set(element, {
            class: highlightClass,
            result: result
        });
    }

    /**
     * Highlight contenteditable div
     */
    function highlightContentEditable(element, highlightClass, result) {
        const text = element.textContent || element.innerText;
        const range = document.createRange();
        range.selectNodeContents(element);
        
        const span = document.createElement('span');
        span.className = highlightClass;
        span.title = `Hate Speech Detected (${(result.hate_probability * 100).toFixed(1)}% confidence)`;
        span.textContent = text;
        
        element.innerHTML = '';
        element.appendChild(span);
    }

    /**
     * Highlight input/textarea field
     */
    function highlightInputField(element, highlightClass, result) {
        // Add class to element itself
        element.classList.add(highlightClass);
        element.title = `Hate Speech Detected (${(result.hate_probability * 100).toFixed(1)}% confidence)`;
        
        // Add visual indicator
        if (!element.parentElement.querySelector('.hate-speech-indicator')) {
            const indicator = document.createElement('div');
            indicator.className = 'hate-speech-indicator';
            indicator.textContent = '⚠️';
            indicator.title = `Hate Speech: ${(result.hate_probability * 100).toFixed(1)}%`;
            element.parentElement.style.position = 'relative';
            element.parentElement.appendChild(indicator);
        }
    }

    /**
     * Remove highlights from element
     */
    function removeHighlights(element) {
        element.classList.remove(CONFIG.HIGHLIGHT_CLASS, CONFIG.BORDERLINE_CLASS);
        element.removeAttribute('title');
        
        const indicator = element.parentElement?.querySelector('.hate-speech-indicator');
        if (indicator) {
            indicator.remove();
        }

        activeHighlights.delete(element);
    }

    /**
     * Remove all highlights
     */
    function removeAllHighlights() {
        document.querySelectorAll(`.${CONFIG.HIGHLIGHT_CLASS}, .${CONFIG.BORDERLINE_CLASS}`).forEach(el => {
            el.classList.remove(CONFIG.HIGHLIGHT_CLASS, CONFIG.BORDERLINE_CLASS);
        });
        document.querySelectorAll('.hate-speech-indicator').forEach(el => el.remove());
        activeHighlights.clear();
    }

    /**
     * Get text from element
     */
    function getTextFromElement(element) {
        if (element.contentEditable === 'true') {
            return element.textContent || element.innerText || '';
        }
        return element.value || '';
    }

    /**
     * Observe new elements added to DOM
     */
    function observeNewElements() {
        const observer = new MutationObserver((mutations) => {
            if (!isEnabled) return;
            
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === 1) { // Element node
                        // Check if it's an input field
                        if (node.tagName === 'INPUT' || node.tagName === 'TEXTAREA' || 
                            node.contentEditable === 'true') {
                            setupInputListener(node);
                        }
                        // Check for input fields inside
                        const inputs = node.querySelectorAll?.('input[type="text"], input[type="search"], textarea, [contenteditable="true"]');
                        inputs?.forEach(input => {
                            if (!input.dataset.hateSpeechMonitored) {
                                input.dataset.hateSpeechMonitored = 'true';
                                setupInputListener(input);
                            }
                        });
                        
                        // Check if it's a comment element
                        if (isCommentElement(node)) {
                            scanCommentElement(node);
                        }
                    }
                });
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    /**
     * Check if element is a comment
     */
    function isCommentElement(element) {
        if (!element || typeof element.querySelector !== 'function') return false;
        
        // Common comment selectors for different platforms
        const commentSelectors = [
            '[id*="comment"]',
            '[class*="comment"]',
            '[data-testid*="comment"]',
            'ytd-comment-renderer', // YouTube
            '[data-testid="tweet"]', // Twitter/X
            '.comment', // Generic
            '.Comment', // Reddit
            '[class*="Comment"]'
        ];
        
        // Check if element matches comment selectors
        for (const selector of commentSelectors) {
            try {
                if (element.matches?.(selector) || element.closest?.(selector)) {
                    return true;
                }
            } catch (e) {
                // Invalid selector, skip
            }
        }
        
        // Check if element contains comment-like text
        const text = element.textContent || '';
        if (text.length > 20 && text.length < 5000) {
            // Check for common comment patterns
            const commentPatterns = [
                /reply|comment|response/i,
                /@\w+/,
                /^\s*[A-Z]/
            ];
            return commentPatterns.some(pattern => pattern.test(text));
        }
        
        return false;
    }

    /**
     * Scan existing comments on the page
     */
    function scanExistingComments() {
        if (!isEnabled) return;
        
        // Wait for page to load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                setTimeout(scanExistingComments, 1000);
            });
            return;
        }
        
        // Find all potential comment elements
        const commentSelectors = [
            'ytd-comment-renderer', // YouTube
            '[data-testid="tweet"]', // Twitter/X
            '.comment',
            '.Comment',
            '[id*="comment"]',
            '[class*="comment"]'
        ];
        
        let foundComments = new Set();
        
        commentSelectors.forEach(selector => {
            try {
                const elements = document.querySelectorAll(selector);
                elements.forEach(el => {
                    if (!el.dataset.hateSpeechScanned) {
                        scanCommentElement(el);
                        foundComments.add(el);
                    }
                });
            } catch (e) {
                // Invalid selector
            }
        });
        
        // Also scan any text nodes that look like comments
        const allTextElements = document.querySelectorAll('p, span, div, article, section');
        allTextElements.forEach(el => {
            if (foundComments.has(el)) return;
            const text = el.textContent?.trim() || '';
            if (text.length > 20 && text.length < 5000 && !el.dataset.hateSpeechScanned) {
                // Check if it's likely a comment (has some structure)
                if (el.children.length === 0 || el.children.length < 5) {
                    scanCommentElement(el);
                }
            }
        });
    }

    /**
     * Scan a single comment element for hate speech
     */
    async function scanCommentElement(element) {
        if (!element || element.dataset.hateSpeechScanned) return;
        
        element.dataset.hateSpeechScanned = 'true';
        
        const text = element.textContent?.trim() || '';
        if (!text || text.length < 3) return;
        
        // Quick keyword check first
        if (!hasSuspiciousKeywords(text)) return;
        
        // Check cache
        const cacheKey = text.toLowerCase().trim();
        if (detectionCache.has(cacheKey)) {
            const result = detectionCache.get(cacheKey);
            if (result.is_hate) {
                highlightComment(element, result);
            }
            return;
        }
        
        // Call API
        const result = await detectHateSpeech(text);
        if (!result.error && result.is_hate) {
            detectionCache.set(cacheKey, result);
            highlightComment(element, result);
        }
    }

    /**
     * Highlight a comment element with hate speech
     */
    function highlightComment(element, result) {
        if (element.dataset.hateSpeechHighlighted) return;
        element.dataset.hateSpeechHighlighted = 'true';
        
        // Add highlight class
        element.classList.add('hate-speech-comment');
        
        // Add visual indicator
        const indicator = document.createElement('span');
        indicator.className = 'hate-speech-comment-badge';
        indicator.textContent = '⚠️ Hate Speech';
        indicator.title = `Hate Speech Detected (${(result.hate_probability * 100).toFixed(1)}% confidence)`;
        
        // Try to insert badge at the beginning of comment
        if (element.firstChild) {
            element.insertBefore(indicator, element.firstChild);
        } else {
            element.appendChild(indicator);
        }
        
        // Highlight the text content
        highlightTextInElement(element, result);
    }

    /**
     * Highlight text within an element
     */
    function highlightTextInElement(element, result) {
        const walker = document.createTreeWalker(
            element,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        
        const textNodes = [];
        let node;
        while (node = walker.nextNode()) {
            if (node.textContent.trim().length > 0) {
                textNodes.push(node);
            }
        }
        
        // For each text node, wrap suspicious parts
        textNodes.forEach(textNode => {
            const text = textNode.textContent;
            const parent = textNode.parentNode;
            
            // Check if text contains hate keywords
            const keywords = ['stupid', 'idiot', 'hate', 'kill', 'worthless', 'useless'];
            const hasKeyword = keywords.some(kw => text.toLowerCase().includes(kw));
            
            if (hasKeyword && text.length > 5) {
                // Wrap the entire text node in a highlight span
                const span = document.createElement('span');
                span.className = 'hate-speech-text-highlight';
                span.textContent = text;
                parent.replaceChild(span, textNode);
            }
        });
    }

    /**
     * Remove all comment highlights
     */
    function removeAllCommentHighlights() {
        document.querySelectorAll('.hate-speech-comment').forEach(el => {
            el.classList.remove('hate-speech-comment');
            el.removeAttribute('data-hate-speech-highlighted');
        });
        document.querySelectorAll('.hate-speech-comment-badge').forEach(el => el.remove());
        document.querySelectorAll('.hate-speech-text-highlight').forEach(el => {
            const parent = el.parentNode;
            parent.replaceChild(document.createTextNode(el.textContent), el);
            parent.normalize();
        });
    }

    /**
     * Observe for new comments being added
     */
    function observeNewComments() {
        // Use MutationObserver to watch for new comments
        const commentObserver = new MutationObserver((mutations) => {
            if (!isEnabled) return;
            
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === 1) { // Element node
                        // Check if it's a comment
                        if (isCommentElement(node)) {
                            // Debounce scanning
                            setTimeout(() => scanCommentElement(node), 500);
                        }
                        // Check for comments inside
                        const comments = node.querySelectorAll?.('ytd-comment-renderer, [data-testid="tweet"], .comment, .Comment, [id*="comment"]');
                        comments?.forEach(comment => {
                            if (!comment.dataset.hateSpeechScanned) {
                                setTimeout(() => scanCommentElement(comment), 500);
                            }
                        });
                    }
                });
            });
        });

        commentObserver.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    /**
     * Intercept form submissions
     */
    function interceptFormSubmissions() {
        document.addEventListener('submit', async (e) => {
            if (!isEnabled) return;

            const form = e.target;
            if (form.tagName !== 'FORM') return;

            // Check all input fields in form
            const inputs = form.querySelectorAll('input[type="text"], input[type="search"], textarea, [contenteditable="true"]');
            let hasHateSpeech = false;
            let hateTexts = [];

            for (const input of inputs) {
                const text = getTextFromElement(input);
                if (!text || text.trim().length < 3) continue;

                // Quick check
                if (hasSuspiciousKeywords(text)) {
                    try {
                        const result = await detectHateSpeech(text);
                        if (result.is_hate) {
                            hasHateSpeech = true;
                            hateTexts.push({
                                text: text.substring(0, 50) + (text.length > 50 ? '...' : ''),
                                confidence: (result.hate_probability * 100).toFixed(1)
                            });
                        }
                    } catch (error) {
                        console.error('Detection error:', error);
                    }
                }
            }

            // If hate speech detected, show warning
            if (hasHateSpeech) {
                e.preventDefault();
                showSubmissionWarning(hateTexts, form);
            }
        }, true); // Use capture phase
    }

    /**
     * Show warning before submission
     */
    function showSubmissionWarning(hateTexts, form) {
        const warning = document.createElement('div');
        warning.className = 'hate-speech-warning';
        warning.innerHTML = `
            <div class="hate-speech-warning-content">
                <h3>⚠️ Hate Speech Detected</h3>
                <p>The following content contains hate speech:</p>
                <ul>
                    ${hateTexts.map(ht => `<li>"${ht.text}" (${ht.confidence}% confidence)</li>`).join('')}
                </ul>
                <div class="hate-speech-warning-buttons">
                    <button class="hate-speech-cancel">Edit & Remove</button>
                    <button class="hate-speech-proceed">Post Anyway</button>
                </div>
            </div>
        `;

        document.body.appendChild(warning);

        // Button handlers
        warning.querySelector('.hate-speech-cancel').addEventListener('click', () => {
            warning.remove();
            // Focus on first input with hate speech
            const firstInput = form.querySelector('input[type="text"], textarea, [contenteditable="true"]');
            if (firstInput) firstInput.focus();
        });

        warning.querySelector('.hate-speech-proceed').addEventListener('click', () => {
            warning.remove();
            // Submit form
            form.submit();
        });
    }

})();
