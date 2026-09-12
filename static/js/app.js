/**
 * Data Science Multi-Agent Solver — Frontend Application Core.
 * Modul terstruktur dan bersih (Clean Human Code) untuk mengelola:
 * 1. Alur Chat & Live 4-Agent Activity Tracker (Real-Time SSE Streaming).
 * 2. Tombol Stop / Cancel Request Aktif (AbortController).
 * 3. 5 Tab Workspace Sinkron (Dataset Explorer, Multi-Plot Gallery, ML Arena, Sandbox, Word Report).
 * 4. Multi-Agent Interactive Sandbox (Python Coder, Doc Reader, Supervisor Router).
 * 5. Rendering Notasi Matematika KaTeX & Clipboard Paste (Ctrl+V).
 */

// =========================================================================
// Global State
// =========================================================================
let stagedMediaFiles = [];
let activePlotsList = [];
let currentAbortController = null;
let lastAgentGeneratedCode = "";
let currentSelectedPlotIndex = 0;
let currentSessionId = null;
let isSessionSidebarOpen = true;
let currentLightboxScale = 1.0;

// =========================================================================
// 1. Lifecycle Initialization & Event Listeners
// =========================================================================
window.addEventListener('DOMContentLoaded', async () => {
    lucide.createIcons();
    initClock();
    initClipboardPasteListener();
    await loadSavedConfig();
    await loadSessionsList();
    await loadDatasetPreview();
    await loadGeneratedPlotsView();
    await loadMLMetricsView();
    await loadSandboxFilesDropdown();
});

function initClock() {
    setInterval(() => {
        const clockEl = document.getElementById('liveClock');
        if (clockEl) {
            const now = new Date();
            clockEl.innerText = now.toLocaleTimeString('id-ID') + ' WIB';
        }
    }, 1000);
}

function initClipboardPasteListener() {
    document.addEventListener('paste', async (event) => {
        const items = (event.clipboardData || window.clipboardData).items;
        if (!items) return;

        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            if (item.kind === 'file' && item.type.startsWith('image/')) {
                const blob = item.getAsFile();
                const timestamp = Date.now();
                const pastedFile = new File([blob], `clipboard_${timestamp}.png`, { type: blob.type });
                addStagedMediaFile(pastedFile);
                showToast("📸 Gambar dari Clipboard (Ctrl+V) berhasil dilampirkan!");
            }
        }
    });
}

// =========================================================================
// 2. Notifikasi Toast & Navigasi 5 Tab Workspace
// =========================================================================
function showToast(message) {
    const toast = document.getElementById('toast');
    const msg = document.getElementById('toastMsg');
    if (!toast || !msg) return;

    msg.innerText = message;
    toast.classList.remove('translate-y-20', 'opacity-0');
    toast.classList.add('translate-y-0', 'opacity-100');

    setTimeout(() => {
        toast.classList.add('translate-y-20', 'opacity-0');
        toast.classList.remove('translate-y-0', 'opacity-100');
    }, 3000);
}

function switchTab(tabId) {
    // Sembunyikan seluruh tab-content
    document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
    
    // Tampilkan tab target
    const targetTab = document.getElementById(tabId);
    if (targetTab) targetTab.classList.remove('hidden');

    // Reset seluruh tombol navigasi tab
    const tabButtons = ['btn-tab-data', 'btn-tab-chart', 'btn-tab-ml', 'btn-tab-code', 'btn-tab-report'];
    tabButtons.forEach(btnId => {
        const btn = document.getElementById(btnId);
        if (btn) {
            btn.className = "px-2.5 py-1 text-xs font-semibold rounded-md text-zinc-400 hover:text-zinc-200 transition flex items-center gap-1";
        }
    });

    // Aktifkan gaya tombol yang sedang dipilih
    const activeBtn = document.getElementById(`btn-${tabId}`);
    if (activeBtn) {
        activeBtn.className = "px-2.5 py-1 text-xs font-semibold rounded-md bg-zinc-800 text-cyan-400 shadow-sm transition flex items-center gap-1";
    }

    // Pemicu pemuatan data sesuai tab
    if (tabId === 'tab-chart') {
        loadGeneratedPlotsView();
    } else if (tabId === 'tab-ml') {
        loadMLMetricsView();
    } else if (tabId === 'tab-code') {
        switchSandboxSubTab('coder');
        loadSandboxFilesDropdown();
    }
}

// =========================================================================
// 3. Konfigurasi Modal (API Key & Model Selection)
// =========================================================================
function openUploadModal() {
    const m = document.getElementById('uploadModal');
    m.classList.remove('hidden');
    m.classList.add('flex');
}

function closeUploadModal() {
    const m = document.getElementById('uploadModal');
    m.classList.add('hidden');
    m.classList.remove('flex');
}

function openSettingsModal() {
    const m = document.getElementById('settingsModal');
    m.classList.remove('hidden');
    m.classList.add('flex');
}

function closeSettingsModal() {
    const m = document.getElementById('settingsModal');
    m.classList.add('hidden');
    m.classList.remove('flex');
}

function toggleProviderSettings() {
    const selectProvider = document.getElementById('selectProvider');
    const provider = selectProvider ? selectProvider.value : 'gemini';
    const geminiSection = document.getElementById('geminiSettingsSection');
    const localSection = document.getElementById('localSettingsSection');

    if (provider === 'local') {
        if (geminiSection) geminiSection.classList.add('hidden');
        if (localSection) localSection.classList.remove('hidden');
    } else {
        if (geminiSection) geminiSection.classList.remove('hidden');
        if (localSection) localSection.classList.add('hidden');
    }
    lucide.createIcons();
}

async function detectLocalModels() {
    const statusEl = document.getElementById('localConnectionStatus');
    const baseUrlInput = document.getElementById('inputLocalBaseUrl');
    const baseUrl = baseUrlInput ? baseUrlInput.value.trim() : 'http://localhost:11434/v1';
    if (statusEl) statusEl.innerHTML = '<span class="text-amber-400">⏳ Mengecek server lokal...</span>';

    try {
        const res = await fetch(`/api/local-models?base_url=${encodeURIComponent(baseUrl)}`);
        const data = await res.json();

        if (data.status === 'online' && data.models && data.models.length > 0) {
            statusEl.innerHTML = `<span class="text-emerald-400">✅ Server Online! Ditemukan ${data.models.length} model terpasang.</span>`;
            
            const container = document.getElementById('localModelDatalistContainer');
            if (container) {
                let selectHtml = `<select onchange="document.getElementById('inputLocalModelName').value=this.value" class="w-full bg-zinc-950 border border-zinc-700 rounded-lg p-2 text-xs text-zinc-200 mt-1 font-mono">`;
                selectHtml += `<option value="">-- Pilih Model Terdeteksi --</option>`;
                data.models.forEach(m => {
                    selectHtml += `<option value="${m}">${m}</option>`;
                });
                selectHtml += `</select>`;
                container.innerHTML = selectHtml;
            }
            if (data.models[0] && document.getElementById('inputLocalModelName')) {
                document.getElementById('inputLocalModelName').value = data.models[0];
            }
            showToast(`✅ Terhubung ke Server Local LLM (${data.models.length} model)!`);
        } else if (data.status === 'online') {
            statusEl.innerHTML = '<span class="text-emerald-400">✅ Server Online, tetapi belum ada model terpasang.</span>';
            showToast("Server Local LLM Online!");
        } else {
            statusEl.innerHTML = '<span class="text-red-400">❌ Server Offline. Pastikan Ollama / FreeToken / LM Studio aktif di port tersebut.</span>';
            showToast("⚠️ Server Local LLM tidak terdeteksi.");
        }
    } catch (e) {
        if (statusEl) statusEl.innerHTML = `<span class="text-red-400">❌ Error: ${e.message}</span>`;
    }
}

function updateNavModelBadge(provider, geminiModel, localModel) {
    const badge = document.getElementById('navSelectedModel');
    if (!badge) return;
    if (provider === 'local') {
        badge.innerText = `[Local GPU] ${localModel || 'qwen2.5-coder:7b'}`;
        badge.className = "font-semibold text-cyan-300 font-mono text-[11px]";
    } else {
        badge.innerText = geminiModel || 'gemma-4-31b-it';
        badge.className = "font-semibold text-zinc-200";
    }
}

async function loadSavedConfig() {
    try {
        const res = await fetch('/api/config');
        const data = await res.json();
        const provider = data.provider || 'gemini';
        
        const selectProvider = document.getElementById('selectProvider');
        if (selectProvider) selectProvider.value = provider;

        if (data.api_key && document.getElementById('inputApiKey')) {
            document.getElementById('inputApiKey').value = data.api_key;
        }
        if (data.selected_model && document.getElementById('selectModel')) {
            document.getElementById('selectModel').value = data.selected_model;
        }
        if (data.local_base_url && document.getElementById('inputLocalBaseUrl')) {
            document.getElementById('inputLocalBaseUrl').value = data.local_base_url;
        }
        if (data.local_model_name && document.getElementById('inputLocalModelName')) {
            document.getElementById('inputLocalModelName').value = data.local_model_name;
        }

        toggleProviderSettings();
        updateNavModelBadge(provider, data.selected_model, data.local_model_name);
    } catch (e) {
        console.error("Gagal memuat konfigurasi API:", e);
    }
}

async function saveSettings() {
    const selectProvider = document.getElementById('selectProvider');
    const provider = selectProvider ? selectProvider.value : 'gemini';
    const apiKey = document.getElementById('inputApiKey') ? document.getElementById('inputApiKey').value.trim() : '';
    const modelName = document.getElementById('selectModel') ? document.getElementById('selectModel').value : 'gemma-4-31b-it';
    const localBaseUrl = document.getElementById('inputLocalBaseUrl') ? document.getElementById('inputLocalBaseUrl').value.trim() : 'http://localhost:11434/v1';
    const localModelName = document.getElementById('inputLocalModelName') ? document.getElementById('inputLocalModelName').value.trim() : 'qwen2.5-coder:7b';

    try {
        await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: provider,
                api_key: apiKey,
                selected_model: modelName,
                local_base_url: localBaseUrl || 'http://localhost:11434/v1',
                local_model_name: localModelName || 'qwen2.5-coder:7b',
                local_api_key: 'ollama'
            })
        });
        updateNavModelBadge(provider, modelName, localModelName);
        closeSettingsModal();
        showToast("✅ Konfigurasi Model AI (Cloud/Local) berhasil disimpan!");
    } catch (e) {
        showToast("❌ Gagal menyimpan konfigurasi.");
    }
}

// =========================================================================
// 4. Manajemen Berkas Terunggah & Staged Media
// =========================================================================
async function uploadFilesToServer(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    const statusBox = document.getElementById('uploadStatusBox');
    statusBox.innerText = "⏳ Mengunggah berkas ke server...";
    statusBox.classList.remove('hidden');

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const result = await res.json();
        statusBox.innerText = `✅ Berhasil mengunggah ${result.files.length} berkas!`;
        
        setTimeout(() => {
            closeUploadModal();
            statusBox.classList.add('hidden');
            loadDatasetPreview();
            loadSandboxFilesDropdown();
            showToast(`✅ ${result.files.length} berkas siap dianalisis!`);
        }, 1000);
    } catch (e) {
        statusBox.innerText = "❌ Gagal mengunggah berkas.";
    }
}

function handleMediaAttachment(event) {
    const files = event.target.files;
    if (files && files.length > 0) {
        for (let i = 0; i < files.length; i++) {
            addStagedMediaFile(files[i]);
        }
        showToast(`📎 ${files.length} berkas dilampirkan.`);
    }
}

function addStagedMediaFile(file) {
    stagedMediaFiles.push(file);
    renderStagedMediaBar();
}

function removeStagedMediaFile(index) {
    stagedMediaFiles.splice(index, 1);
    renderStagedMediaBar();
}

function clearStagedMedia() {
    stagedMediaFiles = [];
    const input = document.getElementById('mediaFileInput');
    if (input) input.value = '';
    renderStagedMediaBar();
}

function renderStagedMediaBar() {
    const bar = document.getElementById('stagedMediaBar');
    if (!bar) return;

    if (stagedMediaFiles.length === 0) {
        bar.classList.add('hidden');
        bar.innerHTML = '';
        return;
    }

    bar.classList.remove('hidden');
    let html = `
        <div class="flex items-center gap-2 overflow-x-auto py-1 w-full">
            <span class="text-cyan-400 flex items-center gap-1 font-semibold text-[11px] whitespace-nowrap mr-1">
                <i data-lucide="paperclip" class="w-3.5 h-3.5"></i> ${stagedMediaFiles.length} Lampiran:
            </span>
    `;

    stagedMediaFiles.forEach((file, idx) => {
        const isImage = file.type.startsWith('image/');
        const previewSrc = isImage ? URL.createObjectURL(file) : '';

        html += `
            <div class="flex items-center gap-1.5 bg-zinc-950 border border-zinc-700 rounded-lg p-1 pr-2 text-xs text-zinc-200 shadow-sm flex-shrink-0">
                ${isImage ? `<img src="${previewSrc}" class="w-6 h-6 object-cover rounded" />` : `<i data-lucide="file" class="w-4 h-4 text-cyan-400"></i>`}
                <span class="max-w-[120px] truncate text-[11px] font-mono">${file.name}</span>
                <button onclick="removeStagedMediaFile(${idx})" class="text-zinc-500 hover:text-red-400 ml-1">
                    <i data-lucide="x" class="w-3 h-3"></i>
                </button>
            </div>
        `;
    });

    html += `
            <button onclick="clearStagedMedia()" class="text-zinc-500 hover:text-red-400 text-[11px] whitespace-nowrap ml-auto pl-2">
                Hapus Semua
            </button>
        </div>
    `;

    bar.innerHTML = html;
    lucide.createIcons();
}

// =========================================================================
// 5. Live Dataset Explorer
// =========================================================================
async function loadDatasetPreview() {
    try {
        const res = await fetch('/api/dataset/preview');
        const data = await res.json();

        const tableContainer = document.getElementById('tableContainer');
        if (data.status === 'success') {
            const healthInfo = data.missing_pct > 0 
                ? `Missing: ${data.missing_count} (${data.missing_pct}%)` 
                : 'Data Bersih (0% Missing)';
                
            document.getElementById('tableSubtitle').innerText = `File: ${data.filename} (${data.total_rows} baris x ${data.total_cols} kolom • ${healthInfo})`;
            document.getElementById('tableRowBadge').innerText = `${data.total_rows} Rows`;

            let html = `<table class="w-full text-left text-xs" id="dataTable">
                <thead class="bg-zinc-900/90 text-zinc-300 uppercase tracking-wider font-semibold border-b border-zinc-800 sticky top-0 backdrop-blur-md">
                    <tr>`;

            data.columns.forEach(col => {
                html += `<th class="p-3">${col}</th>`;
            });
            html += `</tr></thead><tbody class="divide-y divide-zinc-800/60 text-zinc-300 font-mono">`;

            data.rows.forEach(row => {
                html += `<tr class="hover:bg-zinc-800/40 transition">`;
                data.columns.forEach(col => {
                    html += `<td class="p-3">${row[col]}</td>`;
                });
                html += `</tr>`;
            });

            html += `</tbody></table>`;
            tableContainer.innerHTML = html;

            // Render Smart Auto-ML Advisor Banner (EDA-to-ML Handover)
            const advisorCard = document.getElementById('mlAdvisorCard');
            const advisorBadge = document.getElementById('mlAdvisorBadge');
            const advisorReason = document.getElementById('mlAdvisorReason');
            
            if (data.ml_recommendation && advisorCard) {
                advisorCard.classList.remove('hidden');
                if (advisorBadge) advisorBadge.innerText = data.ml_recommendation.task;
                if (advisorReason) {
                    advisorReason.innerHTML = `<strong>Rekomendasi Model:</strong> <span class="text-purple-300 font-semibold">${data.ml_recommendation.algorithm}</span>.<br><span class="text-zinc-400 text-[11px]">${data.ml_recommendation.reason}</span>`;
                }
                
                const mlSelect = document.getElementById('selectMLAlgorithm');
                if (mlSelect && data.ml_recommendation.recommended_key) {
                    mlSelect.value = data.ml_recommendation.recommended_key;
                }
            }
        }
    } catch (e) {
        console.error("Gagal memuat dataset preview:", e);
    }
}

function filterLiveTable() {
    const filter = document.getElementById("tableSearchInput").value.toUpperCase();
    const table = document.getElementById("dataTable");
    if (!table) return;

    const tr = table.getElementsByTagName("tr");
    for (let i = 1; i < tr.length; i++) {
        const text = tr[i].textContent || tr[i].innerText;
        tr[i].style.display = text.toUpperCase().indexOf(filter) > -1 ? "" : "none";
    }
}

// =========================================================================
// 6. Tombol Stop / Cancel Request & Kontrol Status Chat
// =========================================================================
function setButtonStateToStop(loadingId, timerInterval) {
    const btn = document.getElementById('btnSendChat');
    if (!btn) return;
    btn.className = "p-1.5 ml-2 rounded-lg bg-red-500 hover:bg-red-400 text-white font-bold transition shadow-md shadow-red-500/30 flex items-center justify-center animate-pulse";
    btn.innerHTML = '<i data-lucide="square" class="w-4 h-4 fill-current"></i>';
    btn.title = "Hentikan / Batalkan Permintaan (Stop)";
    btn.onclick = () => cancelChatRequest(loadingId, timerInterval);
    lucide.createIcons();
}

function setButtonStateToSend() {
    const btn = document.getElementById('btnSendChat');
    if (!btn) return;
    btn.className = "p-1.5 ml-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-zinc-950 font-bold transition shadow-md shadow-cyan-500/20 flex items-center justify-center";
    btn.innerHTML = '<i data-lucide="arrow-up" class="w-4 h-4"></i>';
    btn.title = "Kirim Pesan";
    btn.onclick = sendChatMessage;
    lucide.createIcons();
}

function cancelChatRequest(loadingId, timerInterval) {
    if (timerInterval) clearInterval(timerInterval);
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }

    const loader = document.getElementById(loadingId);
    if (loader) {
        loader.innerHTML = `
            <div class="glass-card rounded-xl p-3 border border-amber-500/30 text-xs text-amber-300 flex items-center gap-2 bg-amber-500/10">
                <i data-lucide="ban" class="w-4 h-4 text-amber-400 flex-shrink-0"></i>
                <span>Permintaan eksekusi multi-agent dihentikan oleh pengguna (Cancelled).</span>
            </div>
        `;
        lucide.createIcons();
    }

    setButtonStateToSend();
    showToast("⏹️ Permintaan berhasil dihentikan!");
}

// =========================================================================
// 7. Multi-Session Chat Management (ChatGPT / Antigravity Style)
// =========================================================================
function escapeHtml(str) {
    return (str || '')
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

async function loadSessionsList(targetSelectId = null) {
    try {
        const res = await fetch('/api/sessions');
        const data = await res.json();
        const sessions = data.sessions || [];

        const badge = document.getElementById('sessionCountBadge');
        if (badge) badge.innerText = `${sessions.length} Sesi Tersimpan`;

        const container = document.getElementById('sessionListContainer');
        if (!container) return;

        if (sessions.length === 0) {
            container.innerHTML = '<div class="text-[11px] text-zinc-500 text-center py-4 font-mono">Belum ada sesi. Klik + Sesi Baru.</div>';
            return;
        }

        if (!currentSessionId || !sessions.some(s => s.id === currentSessionId)) {
            currentSessionId = targetSelectId || sessions[0].id;
        } else if (targetSelectId) {
            currentSessionId = targetSelectId;
        }

        let html = '';
        sessions.forEach(s => {
            const isActive = s.id === currentSessionId;
            const activeClass = isActive 
                ? 'active bg-zinc-900/90 text-cyan-300 border-cyan-500/50 shadow-sm' 
                : 'text-zinc-300 hover:bg-zinc-900/50 border-transparent';

            html += `
                <div class="session-item group flex items-center justify-between p-2 rounded-xl border text-xs cursor-pointer ${activeClass}" 
                     onclick="switchChatSession('${s.id}')" title="${escapeHtml(s.title || '')}">
                    <div class="flex items-center gap-2 min-w-0 flex-1 pr-1">
                        <i data-lucide="${isActive ? 'message-circle' : 'message-square'}" class="w-3.5 h-3.5 flex-shrink-0 ${isActive ? 'text-cyan-400' : 'text-zinc-500'}"></i>
                        <div class="min-w-0 flex-1">
                            <div class="font-medium truncate text-[12px] leading-tight">${escapeHtml(s.title || 'Sesi Tanpa Judul')}</div>
                            <div class="text-[10px] text-zinc-500 font-mono mt-0.5">${s.updated_at || ''} • ${s.message_count || 0} pesan</div>
                        </div>
                    </div>
                    <div class="session-item-actions flex items-center gap-1 flex-shrink-0">
                        <button onclick="promptRenameSession('${s.id}', '${escapeHtml(s.title || '')}', event)" class="p-1 text-zinc-500 hover:text-cyan-400 rounded transition" title="Ubah Nama">
                            <i data-lucide="edit-3" class="w-3 h-3"></i>
                        </button>
                        <button onclick="promptDeleteSession('${s.id}', event)" class="p-1 text-zinc-500 hover:text-red-400 rounded transition" title="Hapus Sesi">
                            <i data-lucide="trash-2" class="w-3 h-3"></i>
                        </button>
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
        lucide.createIcons();
        updateActiveSessionIndicator(sessions);

        // Muat pesan dari sesi aktif jika chat container belum terisi
        const chatContainer = document.getElementById('chatContainer');
        if (chatContainer && chatContainer.children.length <= 1) {
            await loadSessionMessages(currentSessionId);
        }
    } catch (e) {
        console.error("Gagal memuat daftar sesi:", e);
    }
}

function updateActiveSessionIndicator(sessions = []) {
    const el = document.getElementById('activeSessionIndicator');
    if (!el) return;
    const current = sessions.find(s => s.id === currentSessionId);
    const title = current ? current.title : 'Sesi Aktif';
    el.innerText = `Sesi: ${title}`;
    el.title = title;
}

async function loadSessionMessages(sessionId) {
    if (!sessionId) return;
    try {
        const res = await fetch(`/api/sessions/${sessionId}`);
        const data = await res.json();
        const session = data.session;

        const chatContainer = document.getElementById('chatContainer');
        if (chatContainer) chatContainer.innerHTML = '';

        if (session && session.messages && session.messages.length > 0) {
            session.messages.forEach(msg => {
                appendMessageToChat(
                    msg.role,
                    msg.content,
                    msg.plots,
                    msg.exported_files,
                    msg.attached_media,
                    msg.activity_logs,
                    msg.total_duration
                );
            });
        } else {
            renderEmptyChatGreeting();
        }
    } catch (e) {
        console.error(`Gagal memuat pesan sesi ${sessionId}:`, e);
    }
}

async function createNewChatSession() {
    try {
        const res = await fetch('/api/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: "Sesi Baru" })
        });
        const data = await res.json();
        if (data.status === 'success' && data.session) {
            currentSessionId = data.session.id;
            renderEmptyChatGreeting();
            await loadSessionsList(currentSessionId);
            showToast("✨ Sesi percakapan baru berhasil dibuat!");
        }
    } catch (e) {
        showToast("❌ Gagal membuat sesi baru.");
    }
}

async function switchChatSession(sessionId) {
    if (sessionId === currentSessionId) return;
    currentSessionId = sessionId;
    await loadSessionMessages(sessionId);
    await loadSessionsList(currentSessionId);
}

function renderEmptyChatGreeting() {
    const chatContainer = document.getElementById('chatContainer');
    if (!chatContainer) return;
    chatContainer.innerHTML = `
        <div class="flex flex-col items-start space-y-1.5">
            <div class="flex items-center gap-2 text-xs text-cyan-400 font-semibold ml-1">
                <span class="w-2 h-2 rounded-full bg-cyan-400"></span>
                AI Multi-Agent System (Enterprise 4-Agent Tier)
            </div>
            <div class="glass-card rounded-2xl rounded-tl-sm p-4 text-xs text-zinc-200 leading-relaxed max-w-[90%] shadow-lg border border-zinc-800">
                <p class="font-semibold text-zinc-100 mb-1">👋 Sesi Percakapan Baru Dimulai!</p>
                <p class="text-zinc-400 mb-2">Sistem siap menerima instruksi data science Anda:</p>
                <ul class="list-disc ml-4 space-y-1 text-zinc-300">
                    <li><strong class="text-amber-400">👑 Supervisor:</strong> Mengatur alur kerja agen otomatis.</li>
                    <li><strong class="text-cyan-400">📄 Doc Reader:</strong> Membaca PDF/Word & analisis gambar visual.</li>
                    <li><strong class="text-emerald-400">📊 Data & Stats:</strong> Pembersihan missing value, outlier IQR, & uji hipotesis.</li>
                    <li><strong class="text-purple-400">🤖 ML Specialist:</strong> Pelatihan Machine Learning & Confusion Matrix.</li>
                </ul>
                <p class="mt-2 text-zinc-400">Ketikkan instruksi atau lampirkan berkas di bawah untuk memulai analisis.</p>
            </div>
        </div>
    `;
    lucide.createIcons();
}

function promptRenameSession(sessionId, currentTitle, event) {
    if (event) event.stopPropagation();
    const modal = document.getElementById('renameSessionModal');
    const input = document.getElementById('inputRenameSessionTitle');
    const hiddenId = document.getElementById('targetRenameSessionId');
    if (modal && input && hiddenId) {
        hiddenId.value = sessionId;
        input.value = currentTitle || '';
        modal.classList.remove('hidden');
        modal.classList.add('flex');
        input.focus();
        lucide.createIcons();
    }
}

function closeRenameSessionModal() {
    const modal = document.getElementById('renameSessionModal');
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
    }
}

async function confirmRenameSession() {
    const input = document.getElementById('inputRenameSessionTitle');
    const hiddenId = document.getElementById('targetRenameSessionId');
    if (!input || !hiddenId) return;

    const sessionId = hiddenId.value;
    const newTitle = input.value.trim();
    if (!newTitle) return;

    try {
        const res = await fetch(`/api/sessions/${sessionId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: newTitle })
        });
        const data = await res.json();
        if (data.status === 'success') {
            closeRenameSessionModal();
            await loadSessionsList(currentSessionId);
            showToast("✏️ Nama sesi berhasil diperbarui!");
        }
    } catch (e) {
        showToast("❌ Gagal mengubah nama sesi.");
    }
}

function promptDeleteSession(sessionId, event) {
    if (event) event.stopPropagation();
    const modal = document.getElementById('deleteSessionModal');
    const hiddenId = document.getElementById('targetDeleteSessionId');
    if (modal && hiddenId) {
        hiddenId.value = sessionId;
        modal.classList.remove('hidden');
        modal.classList.add('flex');
        lucide.createIcons();
    }
}

function closeDeleteSessionModal() {
    const modal = document.getElementById('deleteSessionModal');
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
    }
}

async function confirmDeleteSession() {
    const hiddenId = document.getElementById('targetDeleteSessionId');
    if (!hiddenId) return;
    const sessionId = hiddenId.value;

    try {
        const res = await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.status === 'success') {
            closeDeleteSessionModal();
            if (currentSessionId === sessionId) {
                currentSessionId = null;
            }
            await loadSessionsList();
            if (currentSessionId) {
                await loadSessionMessages(currentSessionId);
            }
            showToast("🗑️ Sesi berhasil dihapus.");
        }
    } catch (e) {
        showToast("❌ Gagal menghapus sesi.");
    }
}

function toggleSessionSidebar() {
    const sidebar = document.getElementById('sessionSidebar');
    const openBtn = document.getElementById('btnOpenSidebar');
    if (!sidebar) return;

    isSessionSidebarOpen = !isSessionSidebarOpen;
    if (isSessionSidebarOpen) {
        sidebar.classList.remove('w-0', 'overflow-hidden', 'hidden');
        sidebar.classList.add('w-60');
        if (openBtn) openBtn.classList.add('hidden');
    } else {
        sidebar.classList.remove('w-60');
        sidebar.classList.add('w-0', 'overflow-hidden', 'hidden');
        if (openBtn) openBtn.classList.remove('hidden');
    }
    lucide.createIcons();
}

async function loadChatHistory() {
    await loadSessionsList();
}

function clearChatHistory() {
    const modal = document.getElementById('confirmResetModal');
    if (modal) {
        modal.classList.remove('hidden');
        modal.classList.add('flex');
        lucide.createIcons();
    } else {
        if (confirm("Apakah Anda yakin ingin mengosongkan percakapan pada sesi ini?")) {
            executeConfirmedReset();
        }
    }
}

function closeConfirmResetModal() {
    const modal = document.getElementById('confirmResetModal');
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
    }
}

async function executeConfirmedReset() {
    closeConfirmResetModal();

    if (currentAbortController) {
        try {
            currentAbortController.abort();
        } catch (e) {}
        currentAbortController = null;
    }

    setButtonStateToSend();

    const chatContainer = document.getElementById('chatContainer');
    if (chatContainer) {
        chatContainer.innerHTML = '';
    }

    stagedMediaFiles = [];
    renderStagedMediaBar();
    const queryInput = document.getElementById('userQueryInput');
    if (queryInput) queryInput.value = '';

    try {
        if (currentSessionId) {
            await fetch(`/api/sessions/${currentSessionId}/clear`, { method: 'POST' });
        } else {
            await fetch('/api/history/clear', { method: 'POST' });
        }
        showToast("🗑️ Percakapan sesi ini berhasil dikosongkan.");
        renderEmptyChatGreeting();
        await loadSessionsList(currentSessionId);
    } catch (e) {
        console.error("Error clear history:", e);
    }
}

// =========================================================================
// 8. Code Block Header & Copy Button Enhancement
// =========================================================================
function enhanceCodeBlocks(container) {
    if (!container) return;
    const preElements = container.querySelectorAll('pre:not([data-enhanced="true"])');
    preElements.forEach(pre => {
        pre.setAttribute('data-enhanced', 'true');
        const code = pre.querySelector('code');
        if (!code) return;

        let lang = 'CODE';
        const classNames = code.className || '';
        const langMatch = classNames.match(/language-([a-zA-Z0-9_\-]+)/);
        if (langMatch) {
            lang = langMatch[1].toUpperCase();
        } else {
            const rawText = code.innerText || '';
            if (rawText.includes('import ') || rawText.includes('def ') || rawText.includes('plt.')) {
                lang = 'PYTHON';
            } else if (rawText.trim().startsWith('{') || rawText.trim().startsWith('[')) {
                lang = 'JSON';
            } else if (rawText.includes('SELECT ') || rawText.includes('FROM ')) {
                lang = 'SQL';
            }
        }

        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';

        const header = document.createElement('div');
        header.className = 'code-block-header';
        header.innerHTML = `
            <span class="code-block-lang flex items-center gap-1.5">
                <i data-lucide="terminal" class="w-3.5 h-3.5 text-cyan-400"></i>
                ${lang}
            </span>
            <button type="button" class="code-copy-btn" title="Salin seluruh kode">
                <i data-lucide="copy" class="w-3 h-3 text-zinc-400"></i>
                <span class="copy-text">Salin Kode</span>
            </button>
        `;

        const copyBtn = header.querySelector('.code-copy-btn');
        copyBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const textToCopy = code.innerText || code.textContent || '';
            if (!textToCopy.trim()) return;

            navigator.clipboard.writeText(textToCopy.trim()).then(() => {
                copyBtn.classList.add('copied');
                copyBtn.innerHTML = `
                    <i data-lucide="check" class="w-3 h-3 text-emerald-400"></i>
                    <span class="copy-text text-emerald-300 font-semibold">Tersalin!</span>
                `;
                lucide.createIcons();
                showToast(`📋 Kode ${lang} berhasil disalin!`);

                setTimeout(() => {
                    copyBtn.classList.remove('copied');
                    copyBtn.innerHTML = `
                        <i data-lucide="copy" class="w-3 h-3 text-zinc-400"></i>
                        <span class="copy-text">Salin Kode</span>
                    `;
                    lucide.createIcons();
                }, 2000);
            }).catch(() => {
                fallbackCopyText(textToCopy.trim(), copyBtn);
            });
        });

        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(header);
        wrapper.appendChild(pre);
    });
    lucide.createIcons();
}

// =========================================================================
// 9. Plot Lightbox Zoom (Interactive Fullscreen Click-to-Enlarge)
// =========================================================================
function openPlotLightbox(url, title) {
    const modal = document.getElementById('plotLightboxModal');
    const img = document.getElementById('lightboxPlotImg');
    const titleEl = document.getElementById('lightboxPlotTitle');
    const downloadBtn = document.getElementById('lightboxDownloadBtn');
    const badge = document.getElementById('lightboxZoomBadge');

    if (!modal || !img) return;

    currentLightboxScale = 1.0;
    img.src = url;
    img.style.transform = 'scale(1)';

    if (titleEl) titleEl.innerText = title || 'Plot Preview';
    if (downloadBtn) {
        downloadBtn.href = url;
        downloadBtn.download = title || 'plot.png';
    }
    if (badge) badge.innerText = '100%';

    modal.classList.remove('hidden');
    modal.classList.add('flex');
    document.addEventListener('keydown', handleLightboxKeyDown);
    lucide.createIcons();
}

function closePlotLightbox() {
    const modal = document.getElementById('plotLightboxModal');
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
    }
    document.removeEventListener('keydown', handleLightboxKeyDown);
}

function zoomLightboxImage(delta) {
    currentLightboxScale = Math.max(0.4, Math.min(3.5, currentLightboxScale + delta));
    const img = document.getElementById('lightboxPlotImg');
    const badge = document.getElementById('lightboxZoomBadge');
    if (img) img.style.transform = `scale(${currentLightboxScale.toFixed(2)})`;
    if (badge) badge.innerText = `${Math.round(currentLightboxScale * 100)}%`;
}

function resetLightboxZoom() {
    currentLightboxScale = 1.0;
    const img = document.getElementById('lightboxPlotImg');
    const badge = document.getElementById('lightboxZoomBadge');
    if (img) img.style.transform = 'scale(1)';
    if (badge) badge.innerText = '100%';
}

function handleLightboxKeyDown(e) {
    if (e.key === 'Escape') {
        closePlotLightbox();
    } else if (e.key === '+' || e.key === '=') {
        zoomLightboxImage(0.2);
    } else if (e.key === '-' || e.key === '_') {
        zoomLightboxImage(-0.2);
    } else if (e.key === '0') {
        resetLightboxZoom();
    }
}

async function sendChatMessage() {
    const queryInput = document.getElementById('userQueryInput');
    const query = queryInput.value.trim();
    if (!query && stagedMediaFiles.length === 0) return;

    const selectProvider = document.getElementById('selectProvider');
    const provider = selectProvider ? selectProvider.value : 'gemini';
    const apiKey = document.getElementById('inputApiKey') ? document.getElementById('inputApiKey').value.trim() : '';
    const localBaseUrl = document.getElementById('inputLocalBaseUrl') ? document.getElementById('inputLocalBaseUrl').value.trim() : 'http://localhost:11434/v1';
    const localModelName = document.getElementById('inputLocalModelName') ? document.getElementById('inputLocalModelName').value.trim() : 'qwen2.5-coder:7b';

    if (provider === 'gemini' && !apiKey) {
        openSettingsModal();
        showToast("⚠️ Silakan masukkan API Key Gemini di pengaturan terlebih dahulu.");
        return;
    }

    // 1. Upload media jika ada
    let uploadedMediaPaths = [];
    let uploadedMediaFilenames = [];
    if (stagedMediaFiles.length > 0) {
        const formData = new FormData();
        stagedMediaFiles.forEach(f => formData.append('files', f));
        
        try {
            const uploadRes = await fetch('/api/upload', { method: 'POST', body: formData });
            const uploadResult = await uploadRes.json();
            if (uploadResult.status === 'success') {
                uploadedMediaPaths = uploadResult.files.map(f => f.filepath);
                uploadedMediaFilenames = uploadResult.files.map(f => f.filename);
            }
        } catch (e) {
            console.error("Gagal mengunggah media:", e);
        }
    }

    // 2. Render pesan pengguna di chat
    appendMessageToChat("user", query || "(Lampiran Media Visual)", [], [], uploadedMediaFilenames);
    queryInput.value = '';
    clearStagedMedia();

    // 3. Render Real-Time Progress Card dengan Stopwatch Timer
    const loadingId = 'loading_' + Date.now();
    const chatContainer = document.getElementById('chatContainer');
    const loadingBubble = document.createElement('div');
    loadingBubble.id = loadingId;
    loadingBubble.className = "flex flex-col items-start space-y-1.5 w-full";
    loadingBubble.innerHTML = `
        <div class="flex items-center gap-2 text-xs text-cyan-400 font-semibold ml-1">
            <span class="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
            <span id="${loadingId}_agent_title">👑 Supervisor Agent</span>
        </div>
        <div class="glass-card rounded-2xl rounded-tl-sm p-4 text-xs text-zinc-200 w-full border border-cyan-500/40 space-y-2.5 shadow-xl bg-zinc-900/90 backdrop-blur-md">
            <div class="flex items-center justify-between border-b border-zinc-800 pb-2">
                <span class="text-[11px] font-semibold uppercase tracking-wider text-cyan-300 flex items-center gap-1.5">
                    <i data-lucide="activity" class="w-3.5 h-3.5 animate-spin"></i>
                    Status Alur Eksekusi 4 Agen:
                </span>
                <span id="${loadingId}_time" class="text-[11px] font-mono text-amber-400 font-bold bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/30">
                    ⏱️ 0.1s
                </span>
            </div>
            <div class="p-2.5 rounded-lg bg-zinc-950/80 border border-zinc-800 font-mono text-[11px] text-zinc-300 leading-relaxed" id="${loadingId}_status_text">
                Menganalisis instruksi pengguna dan merencanakan alur kerja 4 agen...
            </div>
            <div class="space-y-1.5 pt-1 text-[11px]" id="${loadingId}_steps_timeline">
                <div class="flex items-center gap-2 text-zinc-400">
                    <span class="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
                    <span>👑 Supervisor: Membaca konteks instruksi & menentukan spesialis...</span>
                </div>
            </div>
        </div>
    `;
    chatContainer.appendChild(loadingBubble);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    lucide.createIcons();

    // Timer Ticker Frontend
    const startTimeMs = Date.now();
    const timerInterval = setInterval(() => {
        const elapsedSec = ((Date.now() - startTimeMs) / 1000).toFixed(1);
        const timeEl = document.getElementById(`${loadingId}_time`);
        if (timeEl) timeEl.innerText = `⏱️ ${elapsedSec}s`;
    }, 200);

    // Aktifkan Tombol Stop
    currentAbortController = new AbortController();
    setButtonStateToStop(loadingId, timerInterval);

    // 4. Konsumsi Real-Time SSE Stream
    let collectedLogs = [];
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query || "Tolong analisis dataset/dokumen yang dilampirkan.",
                session_id: currentSessionId,
                provider: provider,
                api_key: apiKey,
                local_base_url: localBaseUrl,
                local_model_name: localModelName,
                attached_media: uploadedMediaPaths
            }),
            signal: currentAbortController.signal
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n\n");
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const jsonStr = line.replace("data: ", "").trim();
                    if (!jsonStr) continue;

                    try {
                        const event = JSON.parse(jsonStr);

                        if (event.type === 'status') {
                            const titleEl = document.getElementById(`${loadingId}_agent_title`);
                            if (titleEl) titleEl.innerText = event.agent;

                            const textEl = document.getElementById(`${loadingId}_status_text`);
                            if (textEl && event.message) textEl.innerText = event.message;

                            const timelineEl = document.getElementById(`${loadingId}_steps_timeline`);
                            if (timelineEl && event.message) {
                                const stepItem = document.createElement('div');
                                stepItem.className = "flex items-center gap-2 text-emerald-400 text-[11px]";
                                const durationBadge = event.elapsed ? `<span class="text-[10px] text-zinc-500 font-mono">(${event.elapsed}s)</span>` : '';
                                stepItem.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span> <span>${event.agent}: ${event.message} ${durationBadge}</span>`;
                                timelineEl.appendChild(stepItem);
                            }

                            collectedLogs.push({ agent: event.agent, message: event.message, timestamp: event.timestamp, duration_str: `${event.elapsed || 0}s` });
                            chatContainer.scrollTop = chatContainer.scrollHeight;

                        } else if (event.type === 'complete') {
                            clearInterval(timerInterval);
                            const loader = document.getElementById(loadingId);
                            if (loader) loader.remove();

                            if (event.session_id) {
                                currentSessionId = event.session_id;
                            }

                            // Ekstrak kode Python ke sandbox
                            if (event.response.includes("```python")) {
                                const match = event.response.match(/```python([\s\S]*?)```/);
                                if (match) {
                                    lastAgentGeneratedCode = match[1].trim();
                                    const editor = document.getElementById('sandboxCodeEditor');
                                    if (editor) editor.value = lastAgentGeneratedCode;
                                }
                            }

                            appendMessageToChat("assistant", event.response, event.plots, event.exported_files, [], event.activity_logs || collectedLogs, event.total_duration);
                            
                            loadDatasetPreview();
                            loadGeneratedPlotsView();
                            loadMLMetricsView();
                            loadSandboxFilesDropdown();
                            await loadSessionsList(currentSessionId);
                            showToast(`✅ Alur 4 Agen selesai dalam ${event.total_duration || 0}s!`);
                        } else if (event.type === 'error') {
                            clearInterval(timerInterval);
                            const loader = document.getElementById(loadingId);
                            if (loader) loader.remove();
                            appendMessageToChat("assistant", `❌ Error: ${event.message}`);
                        }
                    } catch (err) {
                        console.error("Gagal parsing SSE JSON:", err);
                    }
                }
            }
        }

    } catch (e) {
        clearInterval(timerInterval);
        if (e.name === 'AbortError') {
            console.log("Request dibatalkan oleh pengguna.");
        } else {
            const loader = document.getElementById(loadingId);
            if (loader) loader.remove();
            appendMessageToChat("assistant", `❌ Error koneksi: ${e.message}`);
        }
    } finally {
        setButtonStateToSend();
        currentAbortController = null;
    }
}

function appendMessageToChat(role, content, plots = [], exportedFiles = [], attachedMedia = [], activityLogs = [], totalDuration = null) {
    const chatContainer = document.getElementById('chatContainer');
    const wrapper = document.createElement('div');

    if (role === 'user') {
        wrapper.className = "flex flex-col items-end space-y-1.5";
        const userMsgId = 'user_prompt_' + Math.random().toString(36).substring(2, 9);

        let mediaHtml = '';
        if (attachedMedia && attachedMedia.length > 0) {
            mediaHtml += '<div class="flex flex-wrap gap-2 mb-2 justify-end">';
            attachedMedia.forEach(m => {
                const fname = m.split(/[\/\\]/).pop();
                const isImg = fname.match(/\.(png|jpg|jpeg|webp)$/i);
                if (isImg) {
                    mediaHtml += `<img src="/api/files/download/${fname}" class="h-24 w-auto rounded-lg border border-zinc-700 object-cover shadow-sm" />`;
                } else {
                    mediaHtml += `<div class="px-2.5 py-1 bg-zinc-900 border border-zinc-700 rounded text-[11px] font-mono text-cyan-300 flex items-center gap-1"><i data-lucide="file" class="w-3.5 h-3.5"></i> ${fname}</div>`;
                }
            });
            mediaHtml += '</div>';
        }

        wrapper.innerHTML = `
            <div class="flex items-center gap-2 text-xs text-zinc-400 mr-1 select-none">
                <span class="font-semibold text-zinc-300">User / Mahasiswa</span>
            </div>
            <div class="max-w-[85%] rounded-2xl rounded-tr-sm bg-zinc-800/90 border border-zinc-700/60 p-3.5 shadow-md group">
                ${mediaHtml}
                <div id="${userMsgId}" class="selectable-text user-msg-content text-sm text-zinc-100 break-words leading-relaxed whitespace-pre-wrap select-text">${content}</div>
                <div class="flex items-center justify-end gap-1.5 mt-2.5 pt-2 border-t border-zinc-700/50 select-none">
                    <button onclick="copyTextToClipboard('${userMsgId}', this)" class="copy-btn inline-flex items-center gap-1.5 text-[11px] text-zinc-400 hover:text-cyan-300 bg-zinc-900/90 hover:bg-zinc-700 border border-zinc-700/70 px-2.5 py-1 rounded-lg transition shadow-sm select-none cursor-pointer" title="Salin Teks Prompt">
                        <i data-lucide="copy" class="w-3.5 h-3.5 text-cyan-400"></i>
                        <span class="copy-label font-medium text-[11px]">Salin Prompt</span>
                    </button>
                </div>
            </div>
        `;
    } else {
        wrapper.className = "flex flex-col items-start space-y-1.5 w-full";
        const assistantMsgId = 'assistant_resp_' + Math.random().toString(36).substring(2, 9);

        let plotHtml = '';
        if (plots && plots.length > 0) {
            plots.forEach(plot => {
                const fname = plot.split(/[\/\\]/).pop();
                plotHtml += `
                    <div class="mt-2.5 p-2.5 rounded-xl bg-zinc-950/80 border border-zinc-800 chat-plot-card group relative select-none" onclick="openPlotLightbox('/api/plots/${fname}', '${fname}')">
                        <div class="flex items-center justify-between pb-1.5 text-xs text-zinc-400 px-1">
                            <span class="font-semibold text-cyan-400 flex items-center gap-1.5">
                                <i data-lucide="bar-chart-2" class="w-3.5 h-3.5"></i> Plot Visual: ${fname}
                            </span>
                            <span class="chat-plot-zoom-hint text-[10px] text-cyan-300 font-semibold bg-cyan-500/10 border border-cyan-500/30 px-2 py-0.5 rounded-md flex items-center gap-1">
                                <i data-lucide="zoom-in" class="w-3 h-3"></i> Klik untuk memperbesar
                            </span>
                        </div>
                        <div class="relative overflow-hidden rounded-lg bg-white/5 border border-zinc-800/80 flex items-center justify-center">
                            <img src="/api/plots/${fname}" alt="${fname}" class="max-h-72 w-full object-contain mx-auto transition-transform duration-200 group-hover:scale-[1.01]" />
                            <div class="absolute inset-0 bg-cyan-500/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none flex items-center justify-center">
                                <span class="p-2 rounded-full bg-zinc-900/90 text-cyan-400 border border-cyan-500/40 shadow-xl">
                                    <i data-lucide="maximize-2" class="w-5 h-5"></i>
                                </span>
                            </div>
                        </div>
                    </div>
                `;
            });
        }

        let downloadHtml = '';
        if (exportedFiles && exportedFiles.length > 0) {
            exportedFiles.forEach(file => {
                const fname = file.split(/[\/\\]/).pop();
                downloadHtml += `
                    <a href="/api/files/download/${fname}" download class="inline-flex items-center gap-1.5 px-3 py-1.5 bg-emerald-500 hover:bg-emerald-400 text-zinc-950 font-bold rounded-lg text-xs transition shadow-sm mt-2">
                        <i data-lucide="download" class="w-3.5 h-3.5"></i> Unduh File: ${fname}
                    </a>
                `;
            });
        }

        // Accordion Detail Log Proses Eksekusi Agen + Badge Durasi
        let logsAccordionHtml = '';
        if (activityLogs && activityLogs.length > 0) {
            let logItemsHtml = activityLogs.map(l => `
                <li class="flex items-start gap-2 py-1 border-b border-zinc-800/60 last:border-0 font-mono text-[11px]">
                    <span class="text-cyan-400 font-semibold flex-shrink-0">[${l.timestamp || '--:--'}]</span>
                    <span class="text-amber-400 font-semibold flex-shrink-0">${l.agent}:</span>
                    <span class="text-zinc-300">${l.message}</span>
                    ${l.duration_str ? `<span class="text-zinc-500 text-[10px] ml-auto">(${l.duration_str})</span>` : ''}
                </li>
            `).join('');

            const durationText = totalDuration ? ` • ⏱️ Total Durasi: ${totalDuration}s` : '';

            logsAccordionHtml = `
                <details class="mt-3 p-2.5 bg-zinc-950/70 border border-zinc-800/80 rounded-xl text-xs group transition">
                    <summary class="cursor-pointer font-semibold text-cyan-400 flex items-center justify-between select-none">
                        <span class="flex items-center gap-1.5">
                            <i data-lucide="layers" class="w-3.5 h-3.5 text-cyan-400"></i>
                            🔍 Log Proses Eksekusi 4 Agen (${activityLogs.length} langkah diselesaikan${durationText})
                        </span>
                        <i data-lucide="chevron-down" class="w-3.5 h-3.5 text-zinc-400 group-open:rotate-180 transition-transform"></i>
                    </summary>
                    <ul class="mt-2.5 pt-2 border-t border-zinc-800/60 space-y-1">
                        ${logItemsHtml}
                    </ul>
                </details>
            `;
        }

        let parsedHtml = content;
        if (typeof marked !== 'undefined' && marked.parse) {
            try {
                parsedHtml = marked.parse(content);
            } catch (e) {
                parsedHtml = content;
            }
        }

        wrapper.innerHTML = `
            <div class="flex items-center gap-2 text-xs text-zinc-400 ml-1 select-none">
                <span class="w-4 h-4 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center text-[10px] font-bold">DS</span>
                <span class="font-semibold text-emerald-400">Data Science Multi-Agent (4 Agents)</span>
                ${totalDuration ? `<span class="text-[10px] text-zinc-500 font-mono bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">⏱️ ${totalDuration}s</span>` : ''}
            </div>
            <div class="w-full glass-card rounded-2xl rounded-tl-sm p-4 text-sm text-zinc-200 space-y-3 shadow-lg border-zinc-800">
                <div id="${assistantMsgId}" class="markdown-body prose prose-invert prose-sm max-w-none leading-relaxed text-zinc-200 selectable-text select-text">
                    ${parsedHtml}
                </div>
                ${plotHtml}
                ${downloadHtml}
                ${logsAccordionHtml}
                <div class="flex items-center justify-end pt-2 border-t border-zinc-800/60 select-none">
                    <button onclick="copyTextToClipboard('${assistantMsgId}', this)" class="copy-btn inline-flex items-center gap-1.5 text-[11px] text-zinc-400 hover:text-cyan-300 bg-zinc-900/80 hover:bg-zinc-800 border border-zinc-800 px-2.5 py-1 rounded-lg transition shadow-sm cursor-pointer select-none" title="Salin Seluruh Laporan">
                        <i data-lucide="copy" class="w-3.5 h-3.5 text-cyan-400"></i>
                        <span class="copy-label font-medium text-[11px]">Salin Laporan</span>
                    </button>
                </div>
            </div>
        `;
    }

    chatContainer.appendChild(wrapper);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    lucide.createIcons();
    applyKaTeX(wrapper);
    enhanceCodeBlocks(wrapper);
}

function applyKaTeX(element) {
    if (!element) return;
    if (typeof renderMathInElement !== 'undefined') {
        try {
            renderMathInElement(element, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\(', right: '\\)', display: false },
                    { left: '\\[', right: '\\]', display: true }
                ],
                throwOnError: false,
                ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
            });
        } catch (e) {
            console.error("KaTeX rendering error:", e);
        }
    } else {
        setTimeout(() => {
            if (typeof renderMathInElement !== 'undefined') {
                try {
                    renderMathInElement(element, {
                        delimiters: [
                            { left: '$$', right: '$$', display: true },
                            { left: '$', right: '$', display: false },
                            { left: '\\(', right: '\\)', display: false },
                            { left: '\\[', right: '\\]', display: true }
                        ],
                        throwOnError: false,
                        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
                    });
                } catch (e) {}
            }
        }, 300);
    }
}

// Helper untuk menyalin teks prompt / pesan chat ke Clipboard
function copyTextToClipboard(elementId, btnElement) {
    const el = document.getElementById(elementId);
    if (!el) return;
    
    const text = el.innerText || el.textContent || '';
    if (!text.trim()) return;

    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text.trim()).then(() => {
            handleCopySuccess(btnElement);
        }).catch(() => {
            fallbackCopyText(text.trim(), btnElement);
        });
    } else {
        fallbackCopyText(text.trim(), btnElement);
    }
}

function fallbackCopyText(text, btnElement) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.position = "fixed";
    textArea.style.left = "-999999px";
    textArea.style.top = "-999999px";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    try {
        document.execCommand('copy');
        handleCopySuccess(btnElement);
    } catch (err) {
        showToast("❌ Gagal menyalin ke clipboard.");
    }
    document.body.removeChild(textArea);
}

function handleCopySuccess(btnElement) {
    showToast("📋 Teks prompt berhasil disalin ke Clipboard!");
    if (btnElement) {
        const label = btnElement.querySelector('.copy-label');
        const icon = btnElement.querySelector('i');
        const originalText = label ? label.innerText : "Salin";
        if (label) label.innerText = "Tersalin! ✓";
        btnElement.classList.add('text-emerald-400', 'border-emerald-500/60', 'bg-emerald-950/40');
        setTimeout(() => {
            if (label) label.innerText = originalText;
            btnElement.classList.remove('text-emerald-400', 'border-emerald-500/60', 'bg-emerald-950/40');
        }, 2000);
    }
}

// =========================================================================
// 8. Sinkronisasi Plot Visual ke Tab Interactive Chart (Multi-Plot Gallery)
// =========================================================================
async function loadGeneratedPlotsView() {
    const container = document.getElementById('agentPlotsContainer');
    const countBadgeText = document.getElementById('chartCountText');
    const subtitleEl = document.getElementById('chartTabSubtitle');
    if (!container) return;

    try {
        const res = await fetch('/api/plots/list');
        const data = await res.json();
        activePlotsList = data.plots || [];

        if (countBadgeText) {
            countBadgeText.innerText = `${activePlotsList.length} Grafik Aktif`;
        }

        if (activePlotsList.length === 0) {
            if (subtitleEl) subtitleEl.innerText = "Belum ada grafik visualisasi. Minta agen di chat untuk membuat grafik.";
            container.innerHTML = `
                <div class="glass-card rounded-xl p-8 border border-zinc-800 flex flex-col items-center justify-center min-h-[400px]" id="agentPlotImageWrapper">
                    <div class="text-center text-zinc-500 text-xs max-w-sm space-y-3">
                        <div class="w-12 h-12 rounded-2xl bg-zinc-900 border border-zinc-800 flex items-center justify-center mx-auto text-cyan-400 shadow-md">
                            <i data-lucide="line-chart" class="w-6 h-6"></i>
                        </div>
                        <p class="font-medium text-zinc-300">Belum Ada Grafik Visualisasi Aktif</p>
                        <p class="text-zinc-500 text-[11px] leading-relaxed">Ketik instruksi di chat seperti: <br><span class="text-cyan-400 font-mono">"buatkan gambar grafik tren antrian"</span> atau <span class="text-cyan-400 font-mono">"visualisasikan distribusi data"</span>.</p>
                    </div>
                </div>
            `;
            lucide.createIcons();
            return;
        }

        if (subtitleEl) {
            subtitleEl.innerText = `Menampilkan ${activePlotsList.length} grafik hasil eksekusi komputasi Data Science Multi-Agent.`;
        }

        if (currentSelectedPlotIndex >= activePlotsList.length) {
            currentSelectedPlotIndex = 0;
        }

        renderMultiPlotGallery();

    } catch (e) {
        console.error("Gagal memuat plot list:", e);
    }
}

function renderMultiPlotGallery() {
    const container = document.getElementById('agentPlotsContainer');
    if (!container || activePlotsList.length === 0) return;

    let selectorHtml = '';
    if (activePlotsList.length > 1) {
        selectorHtml = '<div class="flex items-center gap-2 overflow-x-auto pb-2 border-b border-zinc-800/80">';
        activePlotsList.forEach((p, idx) => {
            const isActive = idx === currentSelectedPlotIndex;
            const activeStyle = isActive 
                ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/50 shadow-sm shadow-cyan-500/20" 
                : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-zinc-200 hover:border-zinc-700";
            
            selectorHtml += `
                <button onclick="displaySinglePlot(${idx})" class="px-3 py-1.5 text-xs font-semibold rounded-lg border transition font-mono flex items-center gap-1.5 whitespace-nowrap ${activeStyle}">
                    <i data-lucide="bar-chart-2" class="w-3.5 h-3.5 ${isActive ? 'text-cyan-400' : 'text-zinc-500'}"></i>
                    <span>Plot ${idx + 1}: ${p.filename}</span>
                </button>
            `;
        });
        selectorHtml += '</div>';
    }

    const plot = activePlotsList[currentSelectedPlotIndex];
    let contentHtml = selectorHtml + `
        <div class="glass-card rounded-xl p-4 border border-zinc-800 space-y-3 shadow-xl" id="activePlotDisplay">
            <div class="flex items-center justify-between pb-2.5 border-b border-zinc-800 text-xs">
                <div class="flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
                    <span class="font-bold text-zinc-100 font-mono flex items-center gap-1.5">
                        <i data-lucide="image" class="w-4 h-4 text-cyan-400"></i>
                        ${plot.filename}
                    </span>
                    <span class="text-[11px] text-zinc-500 font-mono bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
                        Dibuat: ${plot.created_at}
                    </span>
                </div>
                <div class="flex items-center gap-2">
                    <a href="${plot.url}" download class="px-3 py-1.5 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 shadow-sm">
                        <i data-lucide="download" class="w-3.5 h-3.5"></i> Unduh Gambar Resolusi Tinggi
                    </a>
                </div>
            </div>
            <div class="bg-zinc-950/90 rounded-xl p-3 flex items-center justify-center border border-zinc-800/80 min-h-[380px] overflow-hidden">
                <img src="${plot.url}" alt="${plot.filename}" class="rounded-lg max-h-[500px] w-full object-contain shadow-inner" />
            </div>
        </div>
    `;

    container.innerHTML = contentHtml;
    lucide.createIcons();
}

function displaySinglePlot(index) {
    if (!activePlotsList || !activePlotsList[index]) return;
    currentSelectedPlotIndex = index;
    renderMultiPlotGallery();
}

// =========================================================================
// 9. Machine Learning Arena & Model Evaluator Logic
// =========================================================================
async function loadMLMetricsView() {
    const cmWrapper = document.getElementById('cmImageWrapper');
    const featWrapper = document.getElementById('featImageWrapper');
    if (!cmWrapper || !featWrapper) return;

    try {
        const res = await fetch('/api/ml/metrics');
        const data = await res.json();

        // 1. Box 1: Confusion Matrix atau Cluster Scatter atau Actual vs Predicted
        if (data.has_confusion_matrix) {
            cmWrapper.innerHTML = `<img src="${data.confusion_matrix_url}" class="rounded max-h-64 object-contain" />`;
        } else if (data.has_cluster_plot) {
            cmWrapper.innerHTML = `<img src="${data.cluster_plot_url}" class="rounded max-h-64 object-contain" />`;
        } else if (data.has_regression_plot) {
            cmWrapper.innerHTML = `<img src="${data.regression_plot_url}" class="rounded max-h-64 object-contain" />`;
        } else {
            cmWrapper.innerHTML = `<span class="text-zinc-500 text-xs">Minta agen: "Latih model klasifikasi, regresi, atau klastering".</span>`;
        }

        // 2. Box 2: Feature Importance atau PCA Projection atau Anomaly Plot
        if (data.has_feature_importance) {
            featWrapper.innerHTML = `<img src="${data.feature_importance_url}" class="rounded max-h-64 object-contain" />`;
        } else if (data.has_pca_plot) {
            featWrapper.innerHTML = `<img src="${data.pca_plot_url}" class="rounded max-h-64 object-contain" />`;
        } else if (data.has_anomaly_plot) {
            featWrapper.innerHTML = `<img src="${data.anomaly_plot_url}" class="rounded max-h-64 object-contain" />`;
        } else {
            featWrapper.innerHTML = `<span class="text-zinc-500 text-xs">Minta agen: "Tampilkan Feature Importance, PCA 2D, atau Anomaly".</span>`;
        }

        lucide.createIcons();
    } catch (e) {
        console.error("Gagal memuat ML metrics:", e);
    }
}

// =========================================================================
// 10. Multi-Agent Interactive Sandbox Functions
// =========================================================================
function switchSandboxSubTab(tabKey) {
    const key = (tabKey || 'coder').replace('sb-panel-', '').replace('btn-sb-', '').replace('sb-', '');

    // Sembunyikan seluruh subpanel
    document.querySelectorAll('.sandbox-subpanel').forEach(p => p.classList.add('hidden'));

    // Tampilkan subpanel target
    const targetPanel = document.getElementById(`sb-panel-${key}`);
    if (targetPanel) {
        targetPanel.classList.remove('hidden');
    }

    const subBtns = [
        { key: 'coder', id: 'btn-sb-coder', color: 'text-amber-400' },
        { key: 'ml', id: 'btn-sb-ml', color: 'text-purple-400' },
        { key: 'reader', id: 'btn-sb-reader', color: 'text-cyan-400' },
        { key: 'supervisor', id: 'btn-sb-supervisor', color: 'text-emerald-400' }
    ];

    subBtns.forEach(btnInfo => {
        const btn = document.getElementById(btnInfo.id);
        if (btn) {
            if (btnInfo.key === key) {
                btn.className = `px-2.5 py-1 text-xs font-semibold rounded-md bg-zinc-800 ${btnInfo.color} shadow-sm transition flex items-center gap-1`;
            } else {
                btn.className = "px-2.5 py-1 text-xs font-semibold rounded-md text-zinc-400 hover:text-zinc-200 transition flex items-center gap-1";
            }
        }
    });

    if (key === 'reader') {
        loadSandboxFilesDropdown();
    }
}

async function trainMLModelSandbox() {
    const algSelect = document.getElementById('selectMLAlgorithm');
    const testSizeSelect = document.getElementById('selectMLTestSize');
    const terminal = document.getElementById('mlSandboxTerminalOutput');
    const badge = document.getElementById('mlSandboxStatusBadge');
    const btn = document.getElementById('btnTrainML');

    const modelType = algSelect ? algSelect.value : 'random_forest';
    const testSize = testSizeSelect ? testSizeSelect.value : '0.2';

    terminal.innerText = `⏳ Melatih model ${modelType.toUpperCase()} (Train-Test Split: ${Math.round((1-parseFloat(testSize))*100)}%/${Math.round(parseFloat(testSize)*100)}%)...`;
    badge.innerText = "Training...";
    badge.className = "text-purple-400 font-semibold text-[11px] animate-pulse";
    btn.disabled = true;

    try {
        const res = await fetch('/api/sandbox/train-ml-model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_type: modelType, test_size: parseFloat(testSize) })
        });
        const data = await res.json();

        if (data.status === 'success') {
            terminal.innerText = data.output;
            badge.innerText = `Trained (${data.elapsed_seconds}s)`;
            badge.className = "text-emerald-400 font-semibold text-[11px]";

            loadMLMetricsView();
            loadGeneratedPlotsView();
            showToast(`🤖 Model ${modelType.toUpperCase()} berhasil dilatih & dievaluasi!`);
        } else {
            terminal.innerText = `❌ Gagal: ${data.detail || data.output}`;
            badge.innerText = "Failed";
            badge.className = "text-red-400 font-semibold text-[11px]";
        }
    } catch (e) {
        terminal.innerText = `❌ Error koneksi: ${e.message}`;
        badge.innerText = "Error";
        badge.className = "text-red-400 font-semibold text-[11px]";
    } finally {
        btn.disabled = false;
    }
}

async function runPythonSandboxCode() {
    const editor = document.getElementById('sandboxCodeEditor');
    const code = editor.value.trim();
    if (!code) {
        showToast("⚠️ Kode Python di sandbox masih kosong.");
        return;
    }

    const terminal = document.getElementById('sandboxTerminalOutput');
    const badge = document.getElementById('sandboxStatusBadge');
    const btn = document.getElementById('btnRunSandbox');

    terminal.innerText = "⏳ Mengeksekusi script Python di Sandbox Kernel...";
    badge.innerText = "Running...";
    badge.className = "text-amber-400 font-semibold text-[11px] animate-pulse";
    btn.disabled = true;

    try {
        const res = await fetch('/api/sandbox/run-python', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code: code })
        });
        const data = await res.json();

        terminal.innerText = data.output || "Eksekusi selesai tanpa output.";
        badge.innerText = `Selesai (${data.elapsed_seconds}s)`;
        badge.className = "text-emerald-400 font-semibold text-[11px]";

        if (data.new_plots && data.new_plots.length > 0) {
            loadGeneratedPlotsView();
            loadMLMetricsView();
            showToast(`📊 ${data.new_plots.length} Grafik baru berhasil dibuat!`);
        } else {
            showToast("✅ Eksekusi Python Sandbox Berhasil!");
        }
    } catch (e) {
        terminal.innerText = `❌ Gagal mengeksekusi sandbox: ${e.message}`;
        badge.innerText = "Error";
        badge.className = "text-red-400 font-semibold text-[11px]";
    } finally {
        btn.disabled = false;
    }
}

function copySandboxCode() {
    const editor = document.getElementById('sandboxCodeEditor');
    if (editor) {
        navigator.clipboard.writeText(editor.value);
        showToast("📋 Kode Python berhasil disalin ke Clipboard!");
    }
}

function resetSandboxCode() {
    const editor = document.getElementById('sandboxCodeEditor');
    if (editor) {
        if (lastAgentGeneratedCode) {
            editor.value = lastAgentGeneratedCode;
            showToast("🔄 Kode direset ke script terbaru dari Data Analyst / ML Agent.");
        } else {
            editor.value = `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats

# Sandbox Python Interaktif dilengkapi proteksi Self-Healing
`;
            showToast("🔄 Kode direset ke template default.");
        }
    }
}

async function loadSandboxFilesDropdown() {
    const select = document.getElementById('selectSandboxDoc');
    if (!select) return;

    try {
        const res = await fetch('/api/sandbox/files');
        const data = await res.json();
        const files = data.files || [];

        if (files.length === 0) {
            select.innerHTML = '<option value="">-- Belum ada berkas terunggah (Unggah di menu atas) --</option>';
            return;
        }

        let html = '<option value="">-- Pilih Berkas untuk Diinspeksi --</option>';
        files.forEach(f => {
            const icon = f.type === 'document' ? '📄' : (f.type === 'image' ? '🖼️' : '📊');
            html += `<option value="${f.filename}">${icon} ${f.filename} (${f.size_kb} KB)</option>`;
        });
        select.innerHTML = html;
    } catch (e) {
        console.error("Gagal memuat dropdown berkas sandbox:", e);
    }
}

async function inspectSelectedDocument() {
    const select = document.getElementById('selectSandboxDoc');
    const filename = select.value;
    if (!filename) {
        showToast("⚠️ Silakan pilih berkas dokumen terlebih dahulu.");
        return;
    }

    const outputEl = document.getElementById('docExtractedOutput');
    const timeBadge = document.getElementById('docInspectTimeBadge');
    outputEl.innerText = `⏳ Membaca dan mengekstrak teks/tabel dari '${filename}'...`;

    try {
        const res = await fetch('/api/sandbox/inspect-doc', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: filename })
        });
        const data = await res.json();
        if (data.status === 'success') {
            outputEl.innerText = data.extracted_text;
            if (timeBadge) timeBadge.innerText = `Waktu Baca: ${data.elapsed_seconds}s`;
            showToast(`✅ Berhasil mengekstrak ${filename}`);
        } else {
            outputEl.innerText = `❌ Gagal: ${data.message}`;
        }
    } catch (e) {
        outputEl.innerText = `❌ Error: ${e.message}`;
    }
}

async function testSupervisorRoutingLogic() {
    const input = document.getElementById('inputRoutingTestQuery');
    const query = input.value.trim();
    if (!query) {
        showToast("⚠️ Masukkan prompt uji terlebih dahulu.");
        return;
    }

    const agentEl = document.getElementById('routingSelectedAgent');
    const reasonEl = document.getElementById('routingReasonText');
    agentEl.innerText = "⏳ Menganalisis...";

    try {
        const res = await fetch('/api/sandbox/test-routing', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });
        const data = await res.json();

        agentEl.innerText = data.agent_name;
        reasonEl.innerHTML = `<strong>Rasional Keputusan:</strong> ${data.reason}`;
        showToast(`👑 Rute: ${data.agent_name}`);
    } catch (e) {
        agentEl.innerText = "Error";
        reasonEl.innerText = e.message;
    }
}

// =========================================================================
// 11. Ekspor Laporan Akademis & Unduh Dataset
// =========================================================================
async function generateAndDownloadReport() {
    showToast("⏳ Sedang menyusun dokumen Laporan Word (.docx)...");
    try {
        const res = await fetch('/api/report/generate', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
            showToast(`✅ Laporan siap: ${data.filename}`);
            window.location.href = data.download_url;
        }
    } catch (e) {
        showToast("❌ Gagal membuat dokumen laporan Word.");
    }
}

async function downloadCleanedData() {
    try {
        const res = await fetch('/api/dataset/preview');
        const data = await res.json();
        if (data.status === 'success') {
            window.location.href = `/api/files/download/${data.filename}`;
            showToast(`📥 Mengunduh ${data.filename}`);
        } else {
            showToast("Belum ada dataset yang diunduh.");
        }
    } catch (e) {
        showToast("❌ Gagal mengunduh dataset.");
    }
}
