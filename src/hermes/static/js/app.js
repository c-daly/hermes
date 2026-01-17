/**
 * Hermes API Test UI - JavaScript Application
 */

// State
let currentEndpoint = 'llm';
let viewMode = 'formatted';
let lastResponse = null;

/**
 * Make an API call to a Hermes endpoint.
 * @param {string} endpoint - The endpoint path (e.g., '/llm')
 * @param {string} method - HTTP method (GET, POST)
 * @param {Object|null} body - Request body for POST requests
 * @returns {Promise<Object>} - Response data or error object
 */
async function callApi(endpoint, method = 'GET', body = null) {
    const options = {
        method,
        headers: {}
    };

    if (body && method !== 'GET') {
        options.headers['Content-Type'] = 'application/json';
        options.body = JSON.stringify(body);
    }

    try {
        const response = await fetch(endpoint, options);

        // Handle binary responses (audio)
        if (response.headers.get('content-type')?.includes('audio/')) {
            const blob = await response.blob();
            return { success: true, audio: blob };
        }

        const data = await response.json();

        if (!response.ok) {
            return { success: false, error: data.detail || 'Request failed', status: response.status };
        }

        return { success: true, data };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

/**
 * Render the API response in the response area.
 * @param {string} endpoint - Which endpoint was called
 * @param {Object} result - The API result object
 */
function renderResponse(endpoint, result) {
    lastResponse = result;
    const area = document.getElementById('response-area');
    const audioPlayer = document.getElementById('audio-player');
    audioPlayer.classList.add('hidden');

    if (!result.success) {
        area.innerHTML = `
            <div class="text-red-600">
                <p class="font-semibold">Error${result.status ? ` (${result.status})` : ''}</p>
                <p>${result.error}</p>
            </div>
        `;
        return;
    }

    if (viewMode === 'json') {
        renderJson(result.data || result);
        return;
    }

    // Formatted rendering based on endpoint
    switch (endpoint) {
        case 'health':
            renderHealth(result.data);
            break;
        case 'llm':
            renderLlm(result.data);
            break;
        case 'tts':
            renderTts(result);
            break;
        case 'nlp':
            renderNlp(result.data);
            break;
        case 'embed':
            renderEmbed(result.data);
            break;
        default:
            renderJson(result.data);
    }
}

/**
 * Render JSON view.
 */
function renderJson(data) {
    const area = document.getElementById('response-area');
    area.innerHTML = `<pre class="text-sm overflow-auto whitespace-pre-wrap">${JSON.stringify(data, null, 2)}</pre>`;
}

/**
 * Render health response.
 */
function renderHealth(data) {
    const area = document.getElementById('response-area');
    const statusColors = {
        healthy: 'text-green-600',
        degraded: 'text-yellow-600',
        unavailable: 'text-red-600'
    };

    let html = `
        <div class="space-y-4">
            <div class="flex items-center gap-2">
                <span class="font-semibold">Status:</span>
                <span class="${statusColors[data.status] || 'text-gray-600'} font-bold uppercase">${data.status}</span>
            </div>
            <div>
                <span class="font-semibold">Service:</span> ${data.service} v${data.version}
            </div>
    `;

    if (data.dependencies) {
        html += `<div><span class="font-semibold">Dependencies:</span><ul class="ml-4 mt-1">`;
        for (const [name, dep] of Object.entries(data.dependencies)) {
            const color = dep.status === 'healthy' ? 'text-green-600' : 'text-red-600';
            html += `<li><span class="${color}">${name}</span>: ${dep.status}</li>`;
        }
        html += `</ul></div>`;
    }

    if (data.capabilities) {
        html += `<div><span class="font-semibold">Capabilities:</span><ul class="ml-4 mt-1">`;
        for (const [name, status] of Object.entries(data.capabilities)) {
            const color = status === 'available' ? 'text-green-600' : 'text-gray-400';
            html += `<li><span class="${color}">${name}</span>: ${status}</li>`;
        }
        html += `</ul></div>`;
    }

    html += `</div>`;
    area.innerHTML = html;
}

/**
 * Render LLM response.
 */
function renderLlm(data) {
    const area = document.getElementById('response-area');
    const message = data.choices?.[0]?.message?.content || 'No response content';

    let html = `
        <div class="space-y-4">
            <div class="bg-gray-50 rounded p-3">
                <p class="font-semibold text-indigo-600 mb-1">Assistant:</p>
                <p class="whitespace-pre-wrap">${escapeHtml(message)}</p>
            </div>
            <div class="text-sm text-gray-500 flex flex-wrap gap-4">
                <span>Provider: ${data.provider}</span>
                <span>Model: ${data.model}</span>
                ${data.usage ? `<span>Tokens: ${data.usage.total_tokens}</span>` : ''}
            </div>
        </div>
    `;
    area.innerHTML = html;
}

/**
 * Render TTS response (audio).
 */
function renderTts(result) {
    const area = document.getElementById('response-area');
    const audioPlayer = document.getElementById('audio-player');
    const audio = document.getElementById('tts-audio');

    if (result.audio) {
        const url = URL.createObjectURL(result.audio);
        audio.src = url;
        audioPlayer.classList.remove('hidden');
        area.innerHTML = `<p class="text-green-600">Audio generated successfully. Use the player below.</p>`;
    } else {
        area.innerHTML = `<p class="text-red-600">No audio in response.</p>`;
    }
}

/**
 * Render NLP response.
 */
function renderNlp(data) {
    const area = document.getElementById('response-area');
    let html = '<div class="grid gap-4 md:grid-cols-2">';

    if (data.tokens) {
        html += `
            <div class="bg-blue-50 rounded p-3">
                <p class="font-semibold text-blue-700 mb-2">Tokens (${data.tokens.length})</p>
                <div class="flex flex-wrap gap-1">
                    ${data.tokens.map(t => `<span class="px-2 py-1 bg-blue-100 rounded text-sm">${escapeHtml(t)}</span>`).join('')}
                </div>
            </div>
        `;
    }

    if (data.pos_tags) {
        html += `
            <div class="bg-purple-50 rounded p-3">
                <p class="font-semibold text-purple-700 mb-2">POS Tags</p>
                <div class="flex flex-wrap gap-1">
                    ${data.pos_tags.map(p => `<span class="px-2 py-1 bg-purple-100 rounded text-sm">${escapeHtml(p.token)}/<span class="text-purple-600">${p.tag}</span></span>`).join('')}
                </div>
            </div>
        `;
    }

    if (data.lemmas) {
        html += `
            <div class="bg-green-50 rounded p-3">
                <p class="font-semibold text-green-700 mb-2">Lemmas</p>
                <div class="flex flex-wrap gap-1">
                    ${data.lemmas.map(l => `<span class="px-2 py-1 bg-green-100 rounded text-sm">${escapeHtml(l)}</span>`).join('')}
                </div>
            </div>
        `;
    }

    if (data.entities && data.entities.length > 0) {
        html += `
            <div class="bg-yellow-50 rounded p-3">
                <p class="font-semibold text-yellow-700 mb-2">Entities</p>
                <div class="space-y-1">
                    ${data.entities.map(e => `<div class="flex items-center gap-2"><span class="px-2 py-1 bg-yellow-100 rounded text-sm">${escapeHtml(e.text)}</span><span class="text-xs text-yellow-600 font-mono">${e.label}</span></div>`).join('')}
                </div>
            </div>
        `;
    }

    html += '</div>';
    area.innerHTML = html;
}

/**
 * Render embed response.
 */
function renderEmbed(data) {
    const area = document.getElementById('response-area');
    const preview = data.embedding.slice(0, 10).map(n => n.toFixed(4)).join(', ');

    let html = `
        <div class="space-y-4">
            <div class="flex items-center gap-4">
                <div class="bg-indigo-50 rounded p-3">
                    <p class="text-sm text-gray-500">Dimension</p>
                    <p class="text-2xl font-bold text-indigo-600">${data.dimension}</p>
                </div>
                <div class="bg-gray-50 rounded p-3 flex-1">
                    <p class="text-sm text-gray-500">Model</p>
                    <p class="font-semibold">${data.model}</p>
                </div>
            </div>
            <div>
                <p class="text-sm text-gray-500 mb-1">Embedding Preview (first 10 values)</p>
                <code class="block bg-gray-100 p-2 rounded text-sm overflow-x-auto">[${preview}, ...]</code>
            </div>
            <div class="text-sm text-gray-400">
                ID: ${data.embedding_id}
            </div>
        </div>
    `;
    area.innerHTML = html;
}

/**
 * Escape HTML to prevent XSS.
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Select an endpoint and show its form.
 */
function selectEndpoint(endpoint) {
    currentEndpoint = endpoint;

    // Update sidebar buttons
    document.querySelectorAll('.endpoint-btn').forEach(btn => {
        if (btn.dataset.endpoint === endpoint) {
            btn.classList.add('bg-indigo-100', 'text-indigo-700');
        } else {
            btn.classList.remove('bg-indigo-100', 'text-indigo-700');
        }
    });

    // Show correct form
    document.querySelectorAll('.endpoint-form').forEach(form => {
        form.classList.add('hidden');
    });
    document.getElementById(`form-${endpoint}`).classList.remove('hidden');
}

/**
 * Set view mode (formatted or json).
 */
function setViewMode(mode) {
    viewMode = mode;

    // Update buttons
    const btnFormatted = document.getElementById('btn-formatted');
    const btnJson = document.getElementById('btn-json');

    if (mode === 'formatted') {
        btnFormatted.classList.add('bg-indigo-600', 'text-white');
        btnFormatted.classList.remove('bg-gray-200', 'text-gray-700');
        btnJson.classList.remove('bg-indigo-600', 'text-white');
        btnJson.classList.add('bg-gray-200', 'text-gray-700');
    } else {
        btnJson.classList.add('bg-indigo-600', 'text-white');
        btnJson.classList.remove('bg-gray-200', 'text-gray-700');
        btnFormatted.classList.remove('bg-indigo-600', 'text-white');
        btnFormatted.classList.add('bg-gray-200', 'text-gray-700');
    }

    // Re-render if we have a response
    if (lastResponse) {
        renderResponse(currentEndpoint, lastResponse);
    }
}

// --- Endpoint Submit Functions ---

async function checkHealth() {
    updateHealthIndicator('checking');
    const result = await callApi('/health');

    if (result.success && result.data.status === 'healthy') {
        updateHealthIndicator('healthy');
    } else if (result.success && result.data.status === 'degraded') {
        updateHealthIndicator('degraded');
    } else {
        updateHealthIndicator('unhealthy');
    }

    renderResponse('health', result);
}

async function submitHealth() {
    await checkHealth();
}

async function submitLlm() {
    const prompt = document.getElementById('llm-prompt').value.trim();
    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }

    const body = {
        prompt,
        provider: document.getElementById('llm-provider').value,
        temperature: parseFloat(document.getElementById('llm-temperature').value) || 0.7
    };

    const model = document.getElementById('llm-model').value.trim();
    if (model) body.model = model;

    const maxTokens = parseInt(document.getElementById('llm-max-tokens').value);
    if (maxTokens > 0) body.max_tokens = maxTokens;

    setLoading(true);
    const result = await callApi('/llm', 'POST', body);
    setLoading(false);
    renderResponse('llm', result);
}

async function submitTts() {
    const text = document.getElementById('tts-text').value.trim();
    if (!text) {
        alert('Please enter text to synthesize');
        return;
    }

    const body = {
        text,
        voice: document.getElementById('tts-voice').value,
        language: document.getElementById('tts-language').value
    };

    setLoading(true);
    const result = await callApi('/tts', 'POST', body);
    setLoading(false);
    renderResponse('tts', result);
}

async function submitNlp() {
    const text = document.getElementById('nlp-text').value.trim();
    if (!text) {
        alert('Please enter text to analyze');
        return;
    }

    const operations = [];
    if (document.getElementById('nlp-tokenize').checked) operations.push('tokenize');
    if (document.getElementById('nlp-pos').checked) operations.push('pos_tag');
    if (document.getElementById('nlp-lemma').checked) operations.push('lemmatize');
    if (document.getElementById('nlp-ner').checked) operations.push('ner');

    if (operations.length === 0) {
        alert('Please select at least one operation');
        return;
    }

    const body = { text, operations };

    setLoading(true);
    const result = await callApi('/simple_nlp', 'POST', body);
    setLoading(false);
    renderResponse('nlp', result);
}

async function submitEmbed() {
    const text = document.getElementById('embed-text').value.trim();
    if (!text) {
        alert('Please enter text to embed');
        return;
    }

    const body = {
        text,
        model: document.getElementById('embed-model').value || 'default'
    };

    setLoading(true);
    const result = await callApi('/embed_text', 'POST', body);
    setLoading(false);
    renderResponse('embed', result);
}

// --- Helper Functions ---

function updateHealthIndicator(status) {
    const indicator = document.getElementById('health-indicator');
    indicator.classList.remove('bg-gray-400', 'bg-green-500', 'bg-yellow-500', 'bg-red-500', 'animate-pulse');

    switch (status) {
        case 'checking':
            indicator.classList.add('bg-gray-400', 'animate-pulse');
            break;
        case 'healthy':
            indicator.classList.add('bg-green-500');
            break;
        case 'degraded':
            indicator.classList.add('bg-yellow-500');
            break;
        case 'unhealthy':
            indicator.classList.add('bg-red-500');
            break;
    }
}

function setLoading(loading) {
    const area = document.getElementById('response-area');
    if (loading) {
        area.innerHTML = `
            <div class="flex items-center gap-2 text-gray-500">
                <div class="w-4 h-4 border-2 border-gray-300 border-t-indigo-600 rounded-full animate-spin"></div>
                <span>Loading...</span>
            </div>
        `;
    }
}

// Initialize health check on page load
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
});
