/**
 * Multimodal Lecture Video Summarizer — Client-Side JavaScript
 * Handles: Tab switching, drag-and-drop upload, SSE progress streaming,
 *          form submission, and auto-redirect on completion.
 */

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initDropZone();
    initForms();
});


/* ──────────────────────────────────────────────
    Tab Switching
   ────────────────────────────────────────────── */

function initTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;
            // Remove active from all tabs and contents
            document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            // Activate selected
            btn.classList.add('active');
            const content = document.getElementById(`tab-${target}`);
            if (content) content.classList.add('active');
        });
    });
}


/* ──────────────────────────────────────────────
    Drag-and-Drop File Upload
   ────────────────────────────────────────────── */

function initDropZone() {
    const zone = document.getElementById('drop-zone');
    if (!zone) return;

    const fileInput = zone.querySelector('input[type="file"]');
    const fileNameEl = document.getElementById('file-name');

    // Drag events
    ['dragenter', 'dragover'].forEach(event => {
        zone.addEventListener(event, (e) => {
            e.preventDefault();
            zone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(event => {
        zone.addEventListener(event, (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
        });
    });

    zone.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        if (file) {
            fileInput.files = e.dataTransfer.files;
            showFileName(file);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files[0]) {
            showFileName(fileInput.files[0]);
        }
    });

    function showFileName(file) {
        // Validate size (250MB max)
        const maxSize = 250 * 1024 * 1024;
        if (file.size > maxSize) {
            showError('File too large. Maximum size is 250MB.');
            fileInput.value = '';
            return;
        }

        // Validate extension
        const allowed = ['mp4', 'mkv', 'avi', 'mov', 'webm', 'flv', 'wmv'];
        const ext = file.name.split('.').pop().toLowerCase();
        if (!allowed.includes(ext)) {
            showError(`Invalid file type. Allowed: ${allowed.join(', ')}`);
            fileInput.value = '';
            return;
        }

        if (fileNameEl) {
            const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
            fileNameEl.textContent = `${file.name} (${sizeMB} MB)`;
            fileNameEl.style.display = 'block';
        }
    }
}


/* ──────────────────────────────────────────────
    Form Submissions
   ────────────────────────────────────────────── */

function initForms() {
    // URL form
    const urlForm = document.getElementById('url-form');
    if (urlForm) {
        urlForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const urlInput = document.getElementById('youtube-url');
            const url = urlInput.value.trim();
            if (!url) {
                showError('Please enter a YouTube URL.');
                return;
            }
            await startProcessingUrl(url);
        });
    }

    // Upload form
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('video-file');
            if (!fileInput.files[0]) {
                showError('Please select a video file.');
                return;
            }
            await startProcessingUpload(fileInput.files[0]);
        });
    }
}


async function startProcessingUrl(youtubeUrl) {
    setSubmitting(true);
    clearError();

    try {
        const res = await fetch('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ youtube_url: youtubeUrl }),
        });

        const data = await res.json();
        if (!res.ok) {
            showError(data.error || 'Processing failed.');
            setSubmitting(false);
            return;
        }

        showProgressSection(data.job_id);
        connectSSE(data.job_id);
    } catch (err) {
        showError('Network error. Please try again.');
        setSubmitting(false);
    }
}


async function startProcessingUpload(file) {
    setSubmitting(true);
    clearError();

    try {
        // Step 1: Upload file
        const formData = new FormData();
        formData.append('video', file);

        const uploadRes = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });

        const uploadData = await uploadRes.json();
        if (!uploadRes.ok) {
            showError(uploadData.error || 'Upload failed.');
            setSubmitting(false);
            return;
        }

        // Step 2: Start processing
        const processRes = await fetch('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_path: uploadData.video_path,
                job_id: uploadData.job_id,
            }),
        });

        const processData = await processRes.json();
        if (!processRes.ok) {
            showError(processData.error || 'Processing failed.');
            setSubmitting(false);
            return;
        }

        showProgressSection(processData.job_id);
        connectSSE(processData.job_id);
    } catch (err) {
        showError('Network error. Please try again.');
        setSubmitting(false);
    }
}


/* ──────────────────────────────────────────────
    SSE — Real-Time Progress
   ────────────────────────────────────────────── */

function connectSSE(jobId) {
    const eventSource = new EventSource(`/api/progress/${jobId}`);

    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.type === 'keepalive') return;

            updateProgress(data);

            if (data.status === 'completed') {
                eventSource.close();
                setTimeout(() => {
                    window.location.href = `/document/${jobId}`;
                }, 1500);
            }

            if (data.status === 'failed') {
                eventSource.close();
                setSubmitting(false);
                showError(data.error || 'Pipeline failed.');
            }

        } catch (err) {
            console.error('SSE parse error:', err);
        }
    };

    eventSource.onerror = () => {
        eventSource.close();
        // Check final status
        fetch(`/api/status/${jobId}`)
            .then(r => r.json())
            .then(data => {
                if (data.status === 'completed') {
                    window.location.href = `/document/${jobId}`;
                } else if (data.status === 'failed') {
                    showError(data.result?.error || 'Connection lost.');
                    setSubmitting(false);
                }
            })
            .catch(() => {
                showError('Connection lost. Please refresh.');
                setSubmitting(false);
            });
    };
}


function updateProgress(data) {
    // Update progress bar
    const fill = document.getElementById('progress-fill');
    if (fill) fill.style.width = `${data.progress || 0}%`;

    // Update message
    const msg = document.getElementById('progress-message');
    if (msg) {
        msg.textContent = data.message || '';
        msg.className = `progress-message ${data.status === 'failed' ? 'error' : ''}`;
    }

    // Update phase cards
    const phaseCards = document.querySelectorAll('.phase-card');
    phaseCards.forEach(card => {
        const phaseNum = parseInt(card.dataset.phase);
        card.classList.remove('active', 'completed', 'failed');

        if (data.status === 'failed' && phaseNum === data.phase) {
            card.classList.add('failed');
        } else if (phaseNum < data.phase) {
            card.classList.add('completed');
        } else if (phaseNum === data.phase) {
            card.classList.add('active');
        }
    });

    // If completed, show success message
    if (data.status === 'completed') {
        const msg = document.getElementById('progress-message');
        if (msg) {
            msg.textContent = '✓ Complete! Redirecting to document viewer...';
            msg.className = 'progress-message';
            msg.style.color = 'var(--accent-green)';
        }
    }
}


/* ──────────────────────────────────────────────
    UI Helpers
   ────────────────────────────────────────────── */

function showProgressSection(jobId) {
    const section = document.getElementById('progress-section');
    if (section) {
        section.classList.add('active');
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    // Hide the input card
    const inputCard = document.getElementById('input-card');
    if (inputCard) inputCard.style.display = 'none';
}

function setSubmitting(isSubmitting) {
    const buttons = document.querySelectorAll('.submit-btn');
    buttons.forEach(btn => {
        btn.disabled = isSubmitting;
        const label = btn.querySelector('.btn-label');
        const spinner = btn.querySelector('.spinner');
        if (label) label.style.display = isSubmitting ? 'none' : 'inline';
        if (spinner) spinner.style.display = isSubmitting ? 'inline-block' : 'none';
    });
}

function showError(message) {
    let alert = document.getElementById('error-alert');
    if (!alert) {
        alert = document.createElement('div');
        alert.id = 'error-alert';
        alert.className = 'alert alert-error';
        const container = document.querySelector('.container') || document.body;
        container.prepend(alert);
    }
    alert.textContent = message;
    alert.style.display = 'block';
    alert.scrollIntoView({ behavior: 'smooth' });
}

function clearError() {
    const alert = document.getElementById('error-alert');
    if (alert) alert.style.display = 'none';
}
