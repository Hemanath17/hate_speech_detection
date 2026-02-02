/**
 * Content Script for Real-Time Hate Speech Detection
 * Monitors input fields, highlights hate speech, and flags before submission
 */

(function() {
    'use strict';

    // Configuration - load from storage or use defaults
    let CONFIG = {
        API_URL: 'http://localhost:5000/api/detect',
        API_BASE: 'http://localhost:5000',
        HEALTH_URL: 'http://localhost:5000/health',
        DEBOUNCE_DELAY: 500,
        TIMEOUT: 5000,
        HIGHLIGHT_CLASS: 'hate-speech-detected',
        BORDERLINE_CLASS: 'hate-speech-borderline',
        ENABLED_KEY: 'extensionEnabled',
        SENSITIVITY_KEY: 'detectionSensitivity',
        MIN_TEXT_LENGTH: 3,
        MAX_TEXT_LENGTH: 2000
    };
    
    // Load config from storage
    chrome.storage.sync.get(['apiUrl', 'apiBase'], (result) => {
        if (result.apiUrl) {
            CONFIG.API_URL = result.apiUrl;
        }
        if (result.apiBase) {
            CONFIG.API_BASE = result.apiBase;
            CONFIG.HEALTH_URL = `${result.apiBase}/health`;
            CONFIG.API_URL = `${result.apiBase}/api/detect`;
        }
    });

    // State
    let isEnabled = true;
    let debounceTimer = null;
    let detectionCache = new Map();
    let activeHighlights = new Map();
    let apiUnavailableLogged = false; // Track if we've already logged API unavailability
    let lastApiCheck = 0;
    let apiErrorCount = 0;
    const API_CHECK_INTERVAL = 60000; // Check API every 60 seconds (reduced frequency)
    const MAX_ERROR_LOGS = 1; // Only log error once per session

    // Initialize
    init();
    
    // Expose manual rescan function for debugging
    window.hateSpeechDetectorRescan = function() {
        console.log('[Hate Speech Detector] Manual rescan triggered');
        scanExistingComments();
    };
    
    // Expose test highlight function for debugging
    window.hateSpeechDetectorTestHighlight = function() {
        console.log('[Hate Speech Detector] Testing highlight on first comment...');
        const firstComment = document.querySelector('ytd-comment-renderer');
        if (firstComment) {
            const testResult = {
                is_hate: true,
                hate_probability: 0.95,
                confidence: 0.95,
                language: 'eng',
                threshold_used: 0.4
            };
            highlightComment(firstComment, testResult);
            console.log('[Hate Speech Detector] Test highlight applied to:', firstComment);
        } else {
            console.warn('[Hate Speech Detector] No comments found to test');
        }
    };
    
    // Expose force scan with test text
    window.hateSpeechDetectorForceTest = async function(testText = "You are stupid and worthless") {
        console.log('[Hate Speech Detector] Force testing with text:', testText);
        const result = await detectHateSpeech(testText);
        console.log('[Hate Speech Detector] Force test result:', result);
        return result;
    };
    
    // Debug function for Reddit
    window.hateSpeechDetectorDebugReddit = function() {
        console.log('[Hate Speech Detector] === Reddit Debug Info ===');
        console.log('URL:', window.location.href);
        console.log('Hostname:', window.location.hostname);
        
        const selectors = [
            'shreddit-comment',
            'faceplate-tracker[source="comment"]',
            '[data-testid="comment"]',
            '.Comment',
            '[class*="Comment"]'
        ];
        
        selectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            console.log(`Selector "${selector}": ${elements.length} elements`);
            if (elements.length > 0) {
                const first = elements[0];
                console.log(`  First element:`, first);
                console.log(`  Tag name:`, first.tagName);
                console.log(`  Classes:`, first.className);
                console.log(`  Text content (first 100 chars):`, first.textContent?.substring(0, 100));
                console.log(`  Inner HTML (first 200 chars):`, first.innerHTML?.substring(0, 200));
            }
        });
        
        // Check for common Reddit comment text patterns
        const allText = document.body.textContent || '';
        const commentPatterns = [
            /fuck you/i,
            /stupid/i,
            /hate/i
        ];
        
        commentPatterns.forEach(pattern => {
            const matches = allText.match(pattern);
            if (matches) {
                console.log(`Found text pattern "${pattern}":`, matches.length, 'matches');
            }
        });
        
        console.log('[Hate Speech Detector] === End Debug Info ===');
    };

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
        console.log('[Hate Speech Detector] Starting monitoring...');
        
        // Monitor existing input fields
        monitorInputFields();

        // Monitor dynamically added input fields
        observeNewElements();

        // Intercept form submissions
        interceptFormSubmissions();

        // Monitor for new comments (start immediately)
        observeNewComments();

        // Scan existing comments with delay to allow YouTube to load
        // YouTube loads comments dynamically, so we need to wait
        setTimeout(() => {
            scanExistingComments();
        }, 2000);
        
        // Also scan after a longer delay for lazy-loaded content
        setTimeout(() => {
            scanExistingComments();
        }, 5000);
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
     * Quick keyword check (local, fast) - Minimal check, let model decide
     */
    function hasSuspiciousKeywords(text) {
        // Very minimal check - just to avoid scanning obviously neutral content
        // Let the ML model do the actual detection
        if (!text || text.length < CONFIG.MIN_TEXT_LENGTH) return false;
        
        // Only skip if text is clearly neutral (greetings, questions, etc.)
        const clearlyNeutral = [
            'hello', 'hi', 'thanks', 'thank you', 'please', 'how are you',
            'what is', 'can you', 'where is', 'when is', 'why is'
        ];
        const textLower = text.toLowerCase().trim();
        
        // If it's a very short neutral phrase, skip
        if (textLower.length < 10 && clearlyNeutral.some(phrase => textLower.includes(phrase))) {
            return false;
        }
        
        // Otherwise, let the model decide (return true to proceed with API call)
        return true;
    }

    /**
     * Check if API server is available
     */
    async function checkApiHealth() {
        try {
            const healthUrl = CONFIG.HEALTH_URL;
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3000);
            
            const response = await fetch(healthUrl, {
                method: 'GET',
                mode: 'cors',
                credentials: 'omit',
                headers: {
                    'Accept': 'application/json'
                },
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                return false;
            }
            
            const data = await response.json();
            return data.status === 'healthy' && data.model_loaded === true;
        } catch (error) {
            // Network error or timeout
            return false;
        }
    }

    /**
     * Call API to detect hate speech
     */
    async function detectHateSpeech(text) {
        try {
            // Check API health (with retry logic)
            if (!window.apiHealthChecked || (Date.now() - (window.lastHealthCheck || 0)) > 30000) {
                const isHealthy = await checkApiHealth();
                window.apiHealthChecked = true;
                window.apiIsHealthy = isHealthy;
                window.lastHealthCheck = Date.now();
                
                if (!isHealthy) {
                    // Only log ONCE per page load
                    if (apiErrorCount === 0) {
                        console.warn('[Hate Speech Detector] API server is not running.');
                        console.warn('[Hate Speech Detector] Start server: python src/api/app.py');
                        apiErrorCount = 1;
                        apiUnavailableLogged = true;
                        lastApiCheck = Date.now();
                    }
                    return {
                        is_hate: false,
                        confidence: 0.0,
                        hate_probability: 0.0,
                        neutral_probability: 1.0,
                        language: 'eng',
                        error: 'API server not available'
                    };
                } else {
                    // API is healthy
                    if (apiUnavailableLogged) {
                        console.log('[Hate Speech Detector] ✓ API server connected!');
                    }
                    apiUnavailableLogged = false;
                    apiErrorCount = 0;
                }
            } else if (!window.apiIsHealthy) {
                // API was previously checked and found to be unhealthy - retry after 30 seconds
                if ((Date.now() - window.lastHealthCheck) > 30000) {
                    // Retry health check
                    const isHealthy = await checkApiHealth();
                    window.apiIsHealthy = isHealthy;
                    window.lastHealthCheck = Date.now();
                    
                    if (!isHealthy) {
                        return {
                            is_hate: false,
                            confidence: 0.0,
                            hate_probability: 0.0,
                            neutral_probability: 1.0,
                            language: 'eng',
                            error: 'API server not available'
                        };
                    } else {
                        console.log('[Hate Speech Detector] ✓ API server is now available!');
                        apiUnavailableLogged = false;
                        apiErrorCount = 0;
                    }
                } else {
                    // Still unhealthy, return error
                    return {
                        is_hate: false,
                        confidence: 0.0,
                        hate_probability: 0.0,
                        neutral_probability: 1.0,
                        language: 'eng',
                        error: 'API server not available'
                    };
                }
            }

            // Make the detection request with timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), CONFIG.TIMEOUT);
            
            try {
                const response = await fetch(CONFIG.API_URL, {
                    method: 'POST',
                    mode: 'cors',
                    credentials: 'omit',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify({ text: text }),
                    signal: controller.signal
                });
                
                clearTimeout(timeoutId);

                if (!response.ok) {
                    // Server error - mark as unhealthy
                    if (response.status === 404 || response.status === 500) {
                        window.apiIsHealthy = false;
                    }
                    return {
                        is_hate: false,
                        confidence: 0.0,
                        hate_probability: 0.0,
                        neutral_probability: 1.0,
                        language: 'eng',
                        error: `API error: ${response.status}`
                    };
                }

                const result = await response.json();
                
                // Validate result structure
                if (!result || typeof result !== 'object') {
                    return {
                        is_hate: false,
                        confidence: 0.0,
                        hate_probability: 0.0,
                        neutral_probability: 1.0,
                        language: 'eng',
                        error: 'Invalid response from API'
                    };
                }
                
                // API is working, reset error count
                if (apiErrorCount > 0) {
                    apiErrorCount = 0;
                    apiUnavailableLogged = false;
                    window.apiIsHealthy = true;
                }
                
                return result;
            } catch (fetchError) {
                clearTimeout(timeoutId);
                if (fetchError.name === 'AbortError') {
                    // Timeout
                    window.apiIsHealthy = false;
                    return {
                        is_hate: false,
                        confidence: 0.0,
                        hate_probability: 0.0,
                        neutral_probability: 1.0,
                        language: 'eng',
                        error: 'API request timeout'
                    };
                }
                // Re-throw to be caught by outer catch
                throw fetchError;
            }
        } catch (error) {
            // Network error (server not running, CORS blocked, etc.)
            // Only log once - already logged in health check
            // Mark API as unhealthy for future requests
            window.apiIsHealthy = false;
            apiUnavailableLogged = true;
            
            // Return neutral result on error (don't throw)
            return {
                is_hate: false,
                confidence: 0.0,
                hate_probability: 0.0,
                neutral_probability: 1.0,
                language: 'eng',
                error: 'API server not available'
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
                        
                        // Check if it's a comment element (but not video metadata)
                        if (isCommentElement(node) && !isVideoMetadata(node)) {
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
     * Check if element is inside a comment section (platform-specific)
     */
    function isInCommentSection(element) {
        if (!element) return false;
        
        const hostname = window.location.hostname;
        
        // YouTube - must be inside #comments section
        if (hostname.includes('youtube.com') || hostname.includes('youtu.be')) {
            const commentsSection = document.querySelector('#comments, ytd-comments, ytd-item-section-renderer[target-id="comments-section"]');
            if (!commentsSection) return false;
            return commentsSection.contains(element);
        }
        
        // Reddit - must be a shreddit-comment or inside comment thread (NOT posts)
        if (hostname.includes('reddit.com')) {
            // Check if it's a post (exclude posts)
            const isPost = element.closest('shreddit-post, [data-testid="post-container"], article[data-testid="post"]') !== null;
            if (isPost) return false;
            
            // Must be a comment element itself or inside one
            const isComment = element.tagName === 'SHREDDIT-COMMENT' ||
                             element.closest('shreddit-comment') !== null ||
                             element.closest('[data-testid="comment"]') !== null ||
                             element.closest('.Comment') !== null;
            
            return isComment;
        }
        
        // Facebook - must be inside comment section
        if (hostname.includes('facebook.com')) {
            const commentSection = element.closest('[data-testid="UFI2Comment/root"], .UFIComment, [role="article"] [data-testid*="comment"]');
            return commentSection !== null;
        }
        
        // Instagram - must be inside comment section
        if (hostname.includes('instagram.com')) {
            const commentSection = element.closest('ul[role="list"] li, [data-testid*="comment"]');
            return commentSection !== null && !element.closest('article header, article h1');
        }
        
        // Twitter/X - must be a reply, not original tweet
        if (hostname.includes('twitter.com') || hostname.includes('x.com')) {
            // Replies have specific structure, original tweets don't
            const tweet = element.closest('[data-testid="tweet"]');
            if (!tweet) return false;
            const isReply = tweet.querySelector('[data-testid="reply"]') !== null;
            const isInThread = tweet.getAttribute('data-conversation-id') !== tweet.getAttribute('data-tweet-id');
            return isReply || isInThread;
        }
        
        // TikTok - must be inside comment section
        if (hostname.includes('tiktok.com')) {
            const commentSection = element.closest('[data-e2e="comment-item"], .comment-item');
            return commentSection !== null;
        }
        
        // LinkedIn - must be inside comment section
        if (hostname.includes('linkedin.com')) {
            const commentSection = element.closest('.comments-comment-item, [data-testid*="comment"]');
            return commentSection !== null && !element.closest('.feed-shared-update-v2__description');
        }
        
        // Pinterest - must be inside comment section
        if (hostname.includes('pinterest.com')) {
            const commentSection = element.closest('[data-testid*="comment"], .comment');
            return commentSection !== null;
        }
        
        // Tumblr - must be inside comment/reply section
        if (hostname.includes('tumblr.com')) {
            const commentSection = element.closest('.post_notes, .note, [data-testid*="comment"]');
            return commentSection !== null;
        }
        
        // Generic fallback - be very strict
        return false;
    }

    /**
     * Check if element is a comment (strict - only actual comments in comment sections)
     */
    function isCommentElement(element) {
        if (!element || typeof element.querySelector !== 'function') return false;
        
        // Must be inside a comment section
        if (!isInCommentSection(element)) {
            return false;
        }
        
        // NEVER scan video metadata, posts, titles, descriptions
        if (isVideoMetadata(element)) {
            return false;
        }
        
        // Platform-specific strict checks
        const hostname = window.location.hostname;
        
        // YouTube - must be ytd-comment-renderer inside comments section
        if (hostname.includes('youtube.com') || hostname.includes('youtu.be')) {
            return element.tagName === 'YTD-COMMENT-RENDERER' && isInCommentSection(element);
        }
        
        // Reddit - must be shreddit-comment
        if (hostname.includes('reddit.com')) {
            return element.tagName === 'SHREDDIT-COMMENT' || 
                   element.matches('[data-testid="comment"]');
        }
        
        // Facebook - must be inside UFI comment
        if (hostname.includes('facebook.com')) {
            return element.closest('[data-testid="UFI2Comment/root"]') !== null;
        }
        
        // Instagram - must be comment list item
        if (hostname.includes('instagram.com')) {
            return element.closest('ul[role="list"] li') !== null && 
                   !element.closest('article header');
        }
        
        // Twitter/X - must be a reply
        if (hostname.includes('twitter.com') || hostname.includes('x.com')) {
            const tweet = element.closest('[data-testid="tweet"]');
            if (!tweet) return false;
            return tweet.querySelector('[data-testid="reply"]') !== null;
        }
        
        // Generic - very strict, return false to avoid false positives
        return false;
    }

    /**
     * Check if we're on a YouTube video watch page (not search results)
     */
    function isYouTubeWatchPage() {
        // Check if we're on a video watch page
        const isWatchPage = window.location.pathname === '/watch' || 
                           window.location.pathname.startsWith('/watch');
        
        // Also check for video player presence
        const hasVideoPlayer = document.querySelector('ytd-watch-flexy, #movie_player, #player');
        
        // Check for comments section
        const hasCommentsSection = document.querySelector('#comments, ytd-comments, ytd-comments-section-header-renderer');
        
        return isWatchPage && (hasVideoPlayer || hasCommentsSection);
    }

    /**
     * Check if element is a video title, description, or metadata (NOT a comment)
     */
    function isVideoMetadata(element) {
        if (!element) return false;
        
        // Check if element is inside video metadata containers
        const videoContainers = [
            'ytd-video-primary-info-renderer', // Video title/description area
            'ytd-video-secondary-info-renderer', // Video metadata
            'ytd-video-meta-block', // Video metadata block
            'ytd-video-renderer', // Video search result item
            'ytd-grid-video-renderer', // Grid video item
            'ytd-playlist-video-renderer', // Playlist video item
            'ytd-compact-video-renderer', // Compact video item
            '#title', // Video title
            '#description', // Video description
            '.ytd-video-primary-info-renderer', // Video info
            '.ytd-video-secondary-info-renderer' // Secondary info
        ];
        
        for (const selector of videoContainers) {
            if (element.closest?.(selector) || element.matches?.(selector)) {
                return true;
            }
        }
        
        return false;
    }

    /**
     * Scan existing comments on the page - ONLY comment sections
     */
    function scanExistingComments() {
        if (!isEnabled) return;
        
        console.log('[Hate Speech Detector] Scanning comment sections only...');
        
        // Wait for page to load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                setTimeout(scanExistingComments, 2000);
            });
            return;
        }
        
        const hostname = window.location.hostname;
        
        // YouTube - ONLY comments section
        if (hostname.includes('youtube.com') || hostname.includes('youtu.be')) {
            if (!isYouTubeWatchPage()) {
                console.log('[Hate Speech Detector] Not on a video watch page, skipping');
                return;
            }
            
            const commentsSection = document.querySelector('#comments, ytd-comments, ytd-item-section-renderer[target-id="comments-section"]');
            if (!commentsSection) {
                console.log('[Hate Speech Detector] YouTube comments section not found, waiting...');
                setTimeout(scanExistingComments, 2000);
                return;
            }
            
            // ONLY scan comments inside the comments section
            const commentRenderers = commentsSection.querySelectorAll('ytd-comment-renderer');
            console.log(`[Hate Speech Detector] Found ${commentRenderers.length} YouTube comments in comments section`);
            
            commentRenderers.forEach(commentRenderer => {
                if (commentRenderer.dataset.hateSpeechScanned) return;
                if (isVideoMetadata(commentRenderer)) return;
                
                const contentText = commentRenderer.querySelector('#content-text, yt-formatted-string#content-text, #content-text yt-formatted-string');
                if (contentText) {
                    const text = contentText.textContent?.trim() || '';
                    if (text.length >= CONFIG.MIN_TEXT_LENGTH && text.length < CONFIG.MAX_TEXT_LENGTH) {
                        commentRenderer.dataset.hateSpeechScanned = 'true';
                        scanCommentElement(commentRenderer, text);
                    }
                }
            });
            return; // Don't continue to other platforms
        }
        
        // Reddit - ONLY comment threads (not posts)
        if (hostname.includes('reddit.com')) {
            console.log('[Hate Speech Detector] Scanning Reddit comment threads only (not posts)...');
            console.log('[Hate Speech Detector] Current URL:', window.location.href);
            
            // ONLY scan shreddit-comment elements (actual comments, not posts)
            const redditComments = document.querySelectorAll('shreddit-comment');
            console.log(`[Hate Speech Detector] Found ${redditComments.length} shreddit-comment elements`);
            
            if (redditComments.length === 0) {
                // Try alternative selectors for Reddit
                const altComments = document.querySelectorAll('[data-testid="comment"], .Comment, [class*="Comment"]');
                console.log(`[Hate Speech Detector] Found ${altComments.length} alternative comment elements`);
                
                altComments.forEach(comment => {
                    // Make sure it's not a post
                    const isPost = comment.closest('shreddit-post, [data-testid="post-container"]') !== null;
                    if (isPost) {
                        console.log('[Hate Speech Detector] Skipping post element');
                        return;
                    }
                    
                    if (comment.dataset.hateSpeechScanned) return;
                    
                    // Extract comment text
                    let commentText = comment.querySelector('p, .md, [slot="text"]') || comment;
                    const text = commentText ? commentText.textContent?.trim() : comment.textContent?.trim() || '';
                    
                    if (text.length >= CONFIG.MIN_TEXT_LENGTH && text.length < CONFIG.MAX_TEXT_LENGTH) {
                        // Skip metadata
                        const isMetadata = /^(u\/|r\/|\d+\s*(min|hour|day|year)s?\s*ago|Reply|Share|Save|permalink|context|parent|report|give award|crosspost)/i.test(text);
                        const isNumeric = /^\d+[km]?$/.test(text.trim());
                        
                        if (!isMetadata && !isNumeric) {
                            comment.dataset.hateSpeechScanned = 'true';
                            console.log(`[Hate Speech Detector] ✓ Scanning Reddit comment (alt): "${text.substring(0, 60)}..."`);
                            scanCommentElement(comment, text);
                        }
                    }
                });
            } else {
                redditComments.forEach(comment => {
                    if (comment.dataset.hateSpeechScanned) return;
                    
                    // Make sure it's not a post
                    const isPost = comment.closest('shreddit-post, [data-testid="post-container"]') !== null;
                    if (isPost) {
                        console.log('[Hate Speech Detector] Skipping post element');
                        return;
                    }
                    
                    // Extract comment text - try multiple methods
                    let commentText = comment.querySelector('div[slot="text"]') ||
                                    comment.querySelector('p[slot="text"]') ||
                                    comment.querySelector('p') ||
                                    comment.querySelector('.md') ||
                                    comment.querySelector('[data-testid="comment"] p');
                    
                    const text = commentText ? commentText.textContent?.trim() : comment.textContent?.trim() || '';
                    
                    if (text.length >= CONFIG.MIN_TEXT_LENGTH && text.length < CONFIG.MAX_TEXT_LENGTH) {
                        // Skip metadata
                        const isMetadata = /^(u\/|r\/|\d+\s*(min|hour|day|year)s?\s*ago|Reply|Share|Save|permalink|context|parent|report|give award|crosspost)/i.test(text);
                        const isNumeric = /^\d+[km]?$/.test(text.trim());
                        
                        if (!isMetadata && !isNumeric) {
                            comment.dataset.hateSpeechScanned = 'true';
                            console.log(`[Hate Speech Detector] ✓ Scanning Reddit comment: "${text.substring(0, 60)}..."`);
                            scanCommentElement(comment, text);
                        } else {
                            console.log(`[Hate Speech Detector] Skipping metadata: "${text.substring(0, 30)}..."`);
                        }
                    } else {
                        console.log(`[Hate Speech Detector] Text length issue: ${text.length} chars (min: ${CONFIG.MIN_TEXT_LENGTH}, max: ${CONFIG.MAX_TEXT_LENGTH})`);
                    }
                });
            }
            
            console.log(`[Hate Speech Detector] Reddit scan complete`);
            return; // Don't continue to other platforms
        }
        
        // No generic scanning - only platform-specific comment sections
        console.log('[Hate Speech Detector] Comment section scan complete');
    }

    /**
     * Scan a single comment element for hate speech
     */
    async function scanCommentElement(element, textOverride = null) {
        if (!element) return;
        
        // NEVER scan video metadata
        if (isVideoMetadata(element)) {
            console.log('[Hate Speech Detector] Skipping video metadata element');
            return;
        }
        
        // Get text - use override if provided, otherwise extract from element
        let text = textOverride;
        if (!text) {
            // For YouTube, try to get text from content-text element
            if (element.tagName === 'YTD-COMMENT-RENDERER') {
                // Double-check it's not video metadata
                if (isVideoMetadata(element)) {
                    return;
                }
                const contentText = element.querySelector('#content-text, yt-formatted-string#content-text, #content-text yt-formatted-string');
                if (contentText && !isVideoMetadata(contentText)) {
                    text = contentText.textContent?.trim() || '';
                } else {
                    text = element.textContent?.trim() || '';
                }
            } else {
                text = element.textContent?.trim() || '';
            }
        }
        
        if (!text || text.length < CONFIG.MIN_TEXT_LENGTH) return;
        
        // Skip very long texts (likely not a single comment or video description)
        if (text.length > CONFIG.MAX_TEXT_LENGTH) return;
        
        // Check cache
        const cacheKey = text.toLowerCase().trim();
        if (detectionCache.has(cacheKey)) {
            const result = detectionCache.get(cacheKey);
            if (result.is_hate) {
                console.log(`[Hate Speech Detector] Cached detection: "${text.substring(0, 50)}..." - HATE`);
                highlightComment(element, result);
            }
            return;
        }
        
        // Call API (no keyword pre-filter for comments - let API decide)
        console.log(`[Hate Speech Detector] Checking: "${text.substring(0, 50)}..."`);
        const result = await detectHateSpeech(text);
        
        console.log(`[Hate Speech Detector] API Result for "${text.substring(0, 30)}...":`, {
            is_hate: result.is_hate,
            hate_probability: result.hate_probability,
            confidence: result.confidence,
            language: result.language,
            error: result.error
        });
        
        if (result.error) {
            // Error already logged in detectHateSpeech, don't duplicate
            // Don't return - still try to highlight if there's a strong keyword match
            if (!hasSuspiciousKeywords(text)) {
                return;
            }
        }
        
        // Cache the result
        detectionCache.set(cacheKey, result);
        
        // Highlight if hate speech detected OR if strong keywords and some probability
        const shouldHighlight = result.is_hate || (hasSuspiciousKeywords(text) && result.hate_probability > 0.25);
        
        if (shouldHighlight) {
            console.log(`[Hate Speech Detector] HATE SPEECH DETECTED: "${text.substring(0, 50)}..." (${(result.hate_probability * 100).toFixed(1)}% confidence, is_hate: ${result.is_hate})`);
            highlightComment(element, result);
        } else {
            console.log(`[Hate Speech Detector] Not hate speech: "${text.substring(0, 30)}..." (hate_prob: ${(result.hate_probability * 100).toFixed(1)}%)`);
        }
    }

    /**
     * Highlight individual hate words in text
     */
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    function highlightHateWords(textElement, text, hateWords) {
        if (!hateWords || hateWords.length === 0) return;
        
        console.log(`[Hate Speech Detector] Highlighting ${hateWords.length} hate words:`, hateWords.map(w => `"${w.word}"`));
        
        // Sort hate words by position (reverse order for DOM manipulation)
        const sortedWords = [...hateWords].sort((a, b) => b.start - a.start);
        
        try {
            // Use DOM manipulation instead of innerHTML to avoid HTML escaping issues
            // Build nodes from start to end
            const fragment = document.createDocumentFragment();
            let currentIndex = 0;
            
            // Sort by start position (ascending) for proper order
            const sortedByStart = [...hateWords].sort((a, b) => a.start - b.start);
            
            sortedByStart.forEach(wordInfo => {
                // Add text before this word
                if (wordInfo.start > currentIndex) {
                    const beforeText = text.substring(currentIndex, wordInfo.start);
                    if (beforeText) {
                        fragment.appendChild(document.createTextNode(beforeText));
                    }
                }
                
                // Create highlighted span for the hate word
                const span = document.createElement('span');
                span.className = 'hate-word-highlight';
                span.setAttribute('data-hate-word', wordInfo.word);
                span.setAttribute('title', `Hate speech word: ${wordInfo.word}`);
                span.textContent = wordInfo.word;
                fragment.appendChild(span);
                
                currentIndex = wordInfo.end;
            });
            
            // Add remaining text after last word
            if (currentIndex < text.length) {
                const remainingText = text.substring(currentIndex);
                if (remainingText) {
                    fragment.appendChild(document.createTextNode(remainingText));
                }
            }
            
            // If no words were processed, just add the full text
            if (sortedByStart.length === 0) {
                fragment.appendChild(document.createTextNode(text));
            }
            
            // Replace element contents with fragment
            if (textElement && textElement.nodeType === Node.ELEMENT_NODE) {
                // Store original attributes
                const originalAttrs = {
                    class: textElement.className,
                    style: textElement.getAttribute('style'),
                    id: textElement.id
                };
                
                // Clear existing content
                while (textElement.firstChild) {
                    textElement.removeChild(textElement.firstChild);
                }
                
                // Append fragment
                textElement.appendChild(fragment);
                
                // Restore original attributes
                if (originalAttrs.class) textElement.className = originalAttrs.class;
                if (originalAttrs.style) textElement.setAttribute('style', originalAttrs.style);
                if (originalAttrs.id) textElement.id = originalAttrs.id;
                
                console.log('[Hate Speech Detector] ✓ Hate words highlighted using DOM manipulation');
            } else if (textElement && textElement.parentNode) {
                // Text node or other - replace with element
                const wrapper = document.createElement('span');
                wrapper.appendChild(fragment);
                if (textElement.className) wrapper.className = textElement.className;
                textElement.parentNode.replaceChild(wrapper, textElement);
            }
            
        } catch (e) {
            console.error('[Hate Speech Detector] Error highlighting words with DOM:', e);
            
            // Fallback: try innerHTML approach
            try {
                let highlightedText = text;
                const sortedWords = [...hateWords].sort((a, b) => b.start - a.start);
                
                sortedWords.forEach(wordInfo => {
                    const before = highlightedText.substring(0, wordInfo.start);
                    const word = highlightedText.substring(wordInfo.start, wordInfo.end);
                    const after = highlightedText.substring(wordInfo.end);
                    
                    const escapedWord = escapeHtml(word);
                    const highlightedWord = `<span class="hate-word-highlight" data-hate-word="${escapedWord.replace(/"/g, '&quot;')}" title="Hate speech: ${escapedWord}">${escapedWord}</span>`;
                    highlightedText = before + highlightedWord + after;
                });
                
                // Escape remaining text
                const parts = highlightedText.split(/(<span[^>]*>.*?<\/span>)/g);
                highlightedText = parts.map((part, i) => i % 2 === 1 ? part : escapeHtml(part)).join('');
                
                if (textElement && 'innerHTML' in textElement) {
                    textElement.innerHTML = highlightedText;
                }
            } catch (e2) {
                console.error('[Hate Speech Detector] Fallback also failed:', e2);
            }
        }
    }
    
    /**
     * Highlight a comment element with hate speech
     */
    function highlightComment(element, result) {
        if (element.dataset.hateSpeechHighlighted === 'true') {
            return;
        }
        element.dataset.hateSpeechHighlighted = 'true';
        
        console.log('[Hate Speech Detector] Applying highlight to:', element.tagName || element.className);
        
        // If hate words are provided, highlight them individually (especially for Reddit)
        if (result.hate_words && result.hate_words.length > 0) {
            console.log(`[Hate Speech Detector] Found ${result.hate_words.length} hate words to highlight:`, result.hate_words.map(w => w.word));
            
            // Find the text element for Reddit
            if (window.location.hostname.includes('reddit.com')) {
                // Try multiple selectors to find the actual text content element
                let textElement = null;
                const selectors = [
                    'div[slot="text"]',
                    'p[slot="text"]',
                    '[slot="text"]',
                    '.md p',
                    '.md',
                    'p',
                    '[data-testid="comment"] p'
                ];
                
                for (const selector of selectors) {
                    const found = element.querySelector(selector);
                    if (found && found.textContent && found.textContent.trim().length > 0) {
                        textElement = found;
                        break;
                    }
                }
                
                // If no specific text element found, use the element itself
                if (!textElement) {
                    textElement = element;
                }
                
                if (textElement) {
                    // Get the original text content (before any modifications)
                    const originalText = textElement.textContent || textElement.innerText || '';
                    if (originalText && originalText.trim().length > 0) {
                        console.log(`[Hate Speech Detector] Found text element, highlighting hate words in: "${originalText.substring(0, 50)}..."`);
                        highlightHateWords(textElement, originalText, result.hate_words);
                    } else {
                        console.warn('[Hate Speech Detector] No text content found in element');
                    }
                }
            }
        }
        
        // For YouTube comments (ytd-comment-renderer), apply styles directly
        // But ONLY if it's actually a comment, not video metadata
        if (element.tagName === 'YTD-COMMENT-RENDERER' && !isVideoMetadata(element)) {
            console.log('[Hate Speech Detector] Detected YouTube comment renderer, applying styles...');
            
            // Create a wrapper overlay div with highlighting (most reliable method)
            if (!element.querySelector('.hate-speech-overlay')) {
                const overlay = document.createElement('div');
                overlay.className = 'hate-speech-overlay';
                overlay.style.cssText = `
                    position: absolute !important;
                    top: -4px !important;
                    left: -4px !important;
                    right: -4px !important;
                    bottom: -4px !important;
                    border: 4px solid #ff0000 !important;
                    border-left: 8px solid #ff0000 !important;
                    border-radius: 8px !important;
                    background-color: rgba(255, 0, 0, 0.2) !important;
                    pointer-events: none !important;
                    z-index: 9999 !important;
                    box-shadow: 0 0 15px rgba(255, 0, 0, 0.6) !important;
                `;
                
                // Make parent relative if not already
                const parentStyle = window.getComputedStyle(element);
                if (parentStyle.position === 'static') {
                    element.style.setProperty('position', 'relative', 'important');
                }
                
                element.insertBefore(overlay, element.firstChild);
                console.log('[Hate Speech Detector] Overlay div added');
            }
            
            // Apply styles using both setProperty and direct assignment for maximum compatibility
            const styles = {
                'background-color': 'rgba(255, 0, 0, 0.25)',
                'border': '4px solid #ff0000',
                'border-left': '8px solid #ff0000',
                'border-radius': '8px',
                'padding': '12px',
                'margin': '10px 0',
                'box-shadow': '0 0 15px rgba(255, 0, 0, 0.5)',
                'position': 'relative',
                'z-index': '10000'
            };
            
            // Apply using setProperty with important
            Object.entries(styles).forEach(([prop, value]) => {
                element.style.setProperty(prop, value, 'important');
            });
            
            // Also add class for CSS fallback
            element.classList.add('hate-speech-comment');
            
            // Force reflow to ensure styles are applied
            void element.offsetHeight;
            
            console.log('[Hate Speech Detector] Styles applied. Computed background:', window.getComputedStyle(element).backgroundColor);
            
            // Also try to find and style the inner content container
            const commentContent = element.querySelector('#comment-content, #content, #body, ytd-comment-thread-renderer');
            if (commentContent) {
                commentContent.style.setProperty('background-color', 'rgba(255, 0, 0, 0.15)', 'important');
                commentContent.style.setProperty('padding', '8px', 'important');
                commentContent.style.setProperty('border-radius', '4px', 'important');
            }
            
            // Find the comment text area and add badge
            const contentText = element.querySelector('#content-text, yt-formatted-string#content-text, #content-text yt-formatted-string, ytd-expander#content-text');
            const commentHeader = element.querySelector('#header, #header-author, ytd-comment-renderer #header');
            
            // Create and insert badge - make it very visible
            const indicator = document.createElement('div');
            indicator.className = 'hate-speech-comment-badge';
            indicator.innerHTML = '⚠️ <strong>HATE SPEECH DETECTED</strong> (' + (result.hate_probability * 100).toFixed(1) + '% confidence)';
            indicator.style.setProperty('display', 'block', 'important');
            indicator.style.setProperty('background-color', '#ff0000', 'important');
            indicator.style.setProperty('color', 'white', 'important');
            indicator.style.setProperty('padding', '8px 12px', 'important');
            indicator.style.setProperty('border-radius', '4px', 'important');
            indicator.style.setProperty('font-size', '13px', 'important');
            indicator.style.setProperty('font-weight', 'bold', 'important');
            indicator.style.setProperty('margin-bottom', '10px', 'important');
            indicator.style.setProperty('cursor', 'help', 'important');
            indicator.style.setProperty('z-index', '10000', 'important');
            indicator.style.setProperty('position', 'relative', 'important');
            indicator.style.setProperty('box-shadow', '0 2px 8px rgba(0, 0, 0, 0.4)', 'important');
            indicator.style.setProperty('width', '100%', 'important');
            indicator.style.setProperty('box-sizing', 'border-box', 'important');
            indicator.title = `Hate Speech Detected (${(result.hate_probability * 100).toFixed(1)}% confidence, Language: ${result.language || 'unknown'})`;
            
            // Insert badge at the top of the comment - try multiple strategies
            let badgeInserted = false;
            
            // Strategy 1: Insert into the body/content area
            const bodyElement = element.querySelector('#body, #content, #comment-content');
            if (bodyElement) {
                bodyElement.insertBefore(indicator, bodyElement.firstChild);
                badgeInserted = true;
                console.log('[Hate Speech Detector] Badge inserted into body element');
            }
            
            // Strategy 2: Insert after header
            if (!badgeInserted && commentHeader) {
                if (commentHeader.nextSibling) {
                    element.insertBefore(indicator, commentHeader.nextSibling);
                } else {
                    commentHeader.parentElement?.insertBefore(indicator, commentHeader.nextSibling);
                }
                badgeInserted = true;
                console.log('[Hate Speech Detector] Badge inserted after header');
            }
            
            // Strategy 3: Insert before content text
            if (!badgeInserted && contentText && contentText.parentElement) {
                contentText.parentElement.insertBefore(indicator, contentText);
                badgeInserted = true;
                console.log('[Hate Speech Detector] Badge inserted before content text');
            }
            
            // Strategy 4: Insert at the very beginning of the comment renderer
            if (!badgeInserted) {
                const firstChild = element.firstElementChild;
                if (firstChild) {
                    element.insertBefore(indicator, firstChild);
                } else {
                    element.appendChild(indicator);
                }
                badgeInserted = true;
                console.log('[Hate Speech Detector] Badge inserted at beginning of element');
            }
            
            // Highlight the text content itself
            if (contentText) {
                contentText.style.setProperty('background-color', 'rgba(255, 0, 0, 0.3)', 'important');
                contentText.style.setProperty('color', '#000', 'important');
                contentText.style.setProperty('padding', '4px 6px', 'important');
                contentText.style.setProperty('border-radius', '3px', 'important');
                contentText.style.setProperty('font-weight', '500', 'important');
            }
            
            // Also highlight any yt-formatted-string elements
            const formattedStrings = element.querySelectorAll('yt-formatted-string');
            formattedStrings.forEach(str => {
                if (str.textContent && str.textContent.trim().length > 0) {
                    str.style.setProperty('background-color', 'rgba(255, 0, 0, 0.25)', 'important');
                    str.style.setProperty('padding', '2px 4px', 'important');
                    str.style.setProperty('border-radius', '2px', 'important');
                }
            });
        } else {
            // For Reddit and other platforms, use standard highlighting
            // Apply aggressive styles
            element.style.setProperty('background-color', 'rgba(255, 0, 0, 0.25)', 'important');
            element.style.setProperty('border', '4px solid #ff0000', 'important');
            element.style.setProperty('border-left', '8px solid #ff0000', 'important');
            element.style.setProperty('border-radius', '8px', 'important');
            element.style.setProperty('padding', '12px', 'important');
            element.style.setProperty('margin', '10px 0', 'important');
            element.style.setProperty('box-shadow', '0 0 15px rgba(255, 0, 0, 0.5)', 'important');
            element.style.setProperty('position', 'relative', 'important');
            element.classList.add('hate-speech-comment');
            
            // Add badge
            const indicator = document.createElement('div');
            indicator.className = 'hate-speech-comment-badge';
            indicator.innerHTML = '⚠️ <strong>HATE SPEECH DETECTED</strong> (' + (result.hate_probability * 100).toFixed(1) + '% confidence)';
            indicator.style.setProperty('display', 'block', 'important');
            indicator.style.setProperty('background-color', '#ff0000', 'important');
            indicator.style.setProperty('color', 'white', 'important');
            indicator.style.setProperty('padding', '8px 12px', 'important');
            indicator.style.setProperty('border-radius', '4px', 'important');
            indicator.style.setProperty('font-size', '13px', 'important');
            indicator.style.setProperty('font-weight', 'bold', 'important');
            indicator.style.setProperty('margin-bottom', '10px', 'important');
            indicator.style.setProperty('width', '100%', 'important');
            indicator.style.setProperty('box-sizing', 'border-box', 'important');
            indicator.title = `Hate Speech Detected (${(result.hate_probability * 100).toFixed(1)}% confidence)`;
            
            // Insert badge at the beginning
            if (element.firstChild) {
                element.insertBefore(indicator, element.firstChild);
            } else {
                element.appendChild(indicator);
            }
        }
        
        // Verify highlighting was applied
        setTimeout(() => {
            const computedStyle = window.getComputedStyle(element);
            const bgColor = computedStyle.backgroundColor;
            if (bgColor === 'rgba(0, 0, 0, 0)' || bgColor === 'transparent') {
                console.warn('[Hate Speech Detector] Highlight may not be visible - styles may be overridden');
            } else {
                console.log('[Hate Speech Detector] ✓ Highlight applied successfully');
            }
        }, 100);
    }

    /**
     * Highlight text within an element
     */
    function highlightTextInElement(element, result) {
        // For YouTube, target the content-text element specifically
        const contentText = element.querySelector('#content-text, yt-formatted-string#content-text, #content-text yt-formatted-string');
        if (contentText) {
            // Highlight the entire content text
            contentText.style.backgroundColor = 'rgba(255, 0, 0, 0.25)';
            contentText.style.color = '#000';
            contentText.style.padding = '2px 4px';
            contentText.style.borderRadius = '2px';
            contentText.style.fontWeight = '500';
            return;
        }
        
        // Fallback: use tree walker for other elements
        const walker = document.createTreeWalker(
            element,
            NodeFilter.SHOW_TEXT,
            {
                acceptNode: function(node) {
                    // Skip if parent is already highlighted
                    if (node.parentElement?.classList.contains('hate-speech-text-highlight')) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    return node.textContent.trim().length > 0 ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
                }
            },
            false
        );
        
        const textNodes = [];
        let node;
        while (node = walker.nextNode()) {
            textNodes.push(node);
        }
        
        // Highlight all text nodes in the comment
        textNodes.forEach(textNode => {
            const text = textNode.textContent;
            if (text.trim().length > 0) {
                const parent = textNode.parentNode;
                const span = document.createElement('span');
                span.className = 'hate-speech-text-highlight';
                span.textContent = text;
                span.style.cssText = 'background-color: rgba(255, 0, 0, 0.25) !important; color: #000 !important; padding: 2px 4px !important; border-radius: 2px !important; font-weight: 500 !important;';
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
                        // Check for YouTube comments specifically (but not video metadata)
                        if (node.tagName === 'YTD-COMMENT-RENDERER' && !isVideoMetadata(node)) {
                            // Only scan if we're on a watch page
                            if (isYouTubeWatchPage()) {
                                setTimeout(() => {
                                    const contentText = node.querySelector('#content-text, yt-formatted-string#content-text, #content-text yt-formatted-string');
                                    if (contentText && !isVideoMetadata(contentText)) {
                                        const text = contentText.textContent?.trim() || '';
                                        if (text.length >= CONFIG.MIN_TEXT_LENGTH && text.length < CONFIG.MAX_TEXT_LENGTH) {
                                            scanCommentElement(node, text);
                                        }
                                    }
                                }, 500);
                            }
                        }
                        
                        // Check for Reddit comments
                        if (window.location.hostname.includes('reddit.com')) {
                            // Check if node itself is a Reddit comment
                            const isRedditComment = 
                                node.tagName === 'SHREDDIT-COMMENT' ||
                                node.tagName === 'FACEPLATE-TRACKER' ||
                                node.matches?.('[data-testid="comment"], .Comment, shreddit-comment, faceplate-tracker[source="comment"]') ||
                                (node.classList && node.classList.contains('Comment')) ||
                                (node.getAttribute && node.getAttribute('data-testid') === 'comment');
                            
                            // Also check if node contains Reddit comments
                            const containsRedditComments = node.querySelector?.('shreddit-comment, [data-testid="comment"], .Comment');
                            
                            if (isRedditComment || containsRedditComments) {
                                setTimeout(() => {
                                    if (isRedditComment && !node.dataset.hateSpeechScanned) {
                                        // Extract text from the comment node
                                        let commentText = null;
                                        
                                        if (node.tagName === 'SHREDDIT-COMMENT') {
                                            commentText = node.querySelector('div[slot="text"]') ||
                                                        node.querySelector('p') ||
                                                        node.querySelector('.md');
                                        } else if (node.tagName === 'FACEPLATE-TRACKER') {
                                            const commentEl = node.querySelector('shreddit-comment') || node.nextElementSibling;
                                            if (commentEl) {
                                                commentText = commentEl.querySelector('div[slot="text"]') ||
                                                            commentEl.querySelector('p');
                                            }
                                        } else {
                                            commentText = node.querySelector('[data-testid="comment"] p') ||
                                                        node.querySelector('.md p') ||
                                                        node.querySelector('[slot="text"]') ||
                                                        node.querySelector('p') ||
                                                        node;
                                        }
                                        
                                        const text = commentText ? commentText.textContent?.trim() : node.textContent?.trim() || '';
                                        
                                        if (text.length >= CONFIG.MIN_TEXT_LENGTH && text.length < CONFIG.MAX_TEXT_LENGTH) {
                                            const isMetadata = /^(u\/|r\/|\d+\s*(min|hour|day|year)s?\s*ago|Reply|Share|Save)/i.test(text);
                                            if (!isMetadata) {
                                                node.dataset.hateSpeechScanned = 'true';
                                                console.log(`[Hate Speech Detector] New Reddit comment detected: "${text.substring(0, 50)}..."`);
                                                scanCommentElement(node, text);
                                            }
                                        }
                                    }
                                    
                                    // Also scan any Reddit comments inside the node
                                    if (containsRedditComments) {
                                        const redditComments = node.querySelectorAll('shreddit-comment, [data-testid="comment"], .Comment');
                                        redditComments.forEach(comment => {
                                            if (!comment.dataset.hateSpeechScanned) {
                                                let commentText = comment.querySelector('div[slot="text"]') ||
                                                                 comment.querySelector('[data-testid="comment"] p') ||
                                                                 comment.querySelector('.md p') ||
                                                                 comment.querySelector('p') ||
                                                                 comment;
                                                
                                                const text = commentText ? commentText.textContent?.trim() : comment.textContent?.trim() || '';
                                                if (text.length >= CONFIG.MIN_TEXT_LENGTH && text.length < CONFIG.MAX_TEXT_LENGTH) {
                                                    const isMetadata = /^(u\/|r\/|\d+\s*(min|hour|day|year)s?\s*ago|Reply|Share|Save|permalink|context|parent|report|give award|crosspost)/i.test(text);
                                                    const isNumeric = /^\d+[km]?$/.test(text.trim());
                                                    
                                                    if (!isMetadata && !isNumeric) {
                                                        comment.dataset.hateSpeechScanned = 'true';
                                                        console.log(`[Hate Speech Detector] ✓ Scanning nested Reddit comment: "${text.substring(0, 50)}..."`);
                                                        scanCommentElement(comment, text);
                                                    }
                                                }
                                            }
                                        });
                                    }
                                }, 300);
                            }
                        }
                        // No generic comment scanning - only platform-specific comment sections
                    }
                });
            });
        });

        commentObserver.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        console.log('[Hate Speech Detector] Comment observer started');
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
