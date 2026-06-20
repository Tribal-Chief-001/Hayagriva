// ==========================================================================
// Hayagriva SOTA Editorial Client Logic
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
    // Generate a unique session ID for conversation memory
    let sessionId = generateUUID();
    
    // UI Elements
    const systemMode = document.getElementById("system-mode");
    const modeText = document.getElementById("mode-text");
    const modelText = document.getElementById("model-text");
    const clearBtn = document.getElementById("clear-btn");
    const nukeBtn = document.getElementById("nuke-btn");
    const docList = document.getElementById("doc-list");
    const refreshDocsBtn = document.getElementById("refresh-docs-btn");
    const ingestBtn = document.getElementById("ingest-btn");
    const ingestSpinner = document.getElementById("ingest-spinner");
    const chatMessages = document.getElementById("chat-messages");
    const typingIndicator = document.getElementById("typing-indicator");
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const sessionHashEl = document.getElementById("session-hash");
    
    // Dropzone Elements
    const dropzone = document.getElementById("dropzone");
    const fileUploader = document.getElementById("file-uploader");
    
    // Citation Drawer Elements
    const citationOverlay = document.getElementById("citation-overlay");
    const citationDrawer = document.getElementById("citation-drawer");
    const closeDrawerBtn = document.getElementById("close-drawer-btn");
    const citationSourceDoc = document.getElementById("citation-source-doc");
    const citationPage = document.getElementById("citation-page");
    const citationScore = document.getElementById("citation-score");
    const citationText = document.getElementById("citation-text");

    // Initialize Page & Session Monogram
    updateSessionMonogram();
    fetchSystemStatus();
    fetchDocuments();

    // ----------------------------------------------------
    // Drag and Drop Upload Event Listeners
    // ----------------------------------------------------
    
    // Click on dropzone triggers file dialog
    dropzone.addEventListener("click", () => {
        fileUploader.click();
    });

    // Prevent programmatic click from bubbling up to dropzone
    fileUploader.addEventListener("click", (e) => {
        e.stopPropagation();
    });

    // File selected from dialog
    fileUploader.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Drag-over hover effects
    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        }, false);
    });

    // Drop file trigger
    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // ----------------------------------------------------
    // Ingest and Button Event Listeners
    // ----------------------------------------------------
    
    // Ingest Directory Trigger (Local scanning backup)
    ingestBtn.addEventListener("click", async () => {
        ingestBtn.disabled = true;
        const btnTxt = ingestBtn.querySelector(".btn-txt");
        const origBtnText = btnTxt.innerHTML;
        btnTxt.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> SCANNING...';
        ingestSpinner.classList.remove("hidden");

        try {
            const res = await fetch("/api/ingest", { method: "POST" });
            const data = await res.json();

            if (data.status === "success") {
                const n = data.ingested.length;
                const s = data.skipped.length;
                btnTxt.innerHTML = `<i class="fa-solid fa-check"></i> ${n} INDEXED, ${s} SKIPPED`;
                fetchDocuments();
            } else {
                btnTxt.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> SCAN FAILED';
                console.error("Ingest error:", data.message);
            }
        } catch (err) {
            console.error(err);
            btnTxt.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> NETWORK ERROR';
        } finally {
            ingestSpinner.classList.add("hidden");
            setTimeout(() => {
                btnTxt.innerHTML = origBtnText;
                ingestBtn.disabled = false;
            }, 3500);
        }
    });

    // Refresh Documents List
    refreshDocsBtn.addEventListener("click", fetchDocuments);

    // Clear Chat History & reset session
    clearBtn.addEventListener("click", () => {
        if (confirm("Are you sure you want to clear the chat history?")) {
            sessionId = generateUUID();
            updateSessionMonogram();
            
            // Keep welcoming block only
            const welcome = chatMessages.querySelector(".init-msg");
            chatMessages.innerHTML = "";
            if (welcome) {
                chatMessages.appendChild(welcome);
            }
        }
    });

    // Nuke entire database
    if (nukeBtn) {
        nukeBtn.addEventListener("click", async () => {
            if (confirm("WARNING: Are you absolutely sure you want to wipe the ENTIRE vector database? This cannot be undone.")) {
                try {
                    const res = await fetch("/api/purge_db", { method: "POST" });
                    if (res.ok) {
                        alert("Database successfully nuked and rebuilt.");
                        fetchDocuments();
                    } else {
                        const data = await res.json();
                        alert("Failed to nuke database: " + data.message);
                    }
                } catch (e) {
                    alert("Error: " + e.message);
                }
            }
        });
    }

    // Submit Query
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const queryText = userInput.value.trim();
        if (!queryText) return;

        // Clear and lock input
        userInput.value = "";
        userInput.disabled = true;
        sendBtn.disabled = true;

        // Append User Query
        appendMessage("user", queryText);
        
        // Show Typing Indicator
        const thinkingTxt = typingIndicator.querySelector(".thinking-txt");
        thinkingTxt.innerHTML = "Initializing telemetry... <span class='stopwatch'>0.0s</span>";
        typingIndicator.classList.remove("hidden");
        scrollToBottom();

        // Start Live Stopwatch
        const startTime = Date.now();
        const timerInterval = setInterval(() => {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            const stopwatchEl = typingIndicator.querySelector(".stopwatch");
            if (stopwatchEl) {
                stopwatchEl.textContent = elapsed + "s";
            }
        }, 100);

        try {
            await executeStreamingQuery(queryText, timerInterval);
        } catch (err) {
            console.error("Stream error: ", err);
            clearInterval(timerInterval);
            appendMessage("assistant", "Apologies, but an issue occurred while querying the server stream.");
            typingIndicator.classList.add("hidden");
        } finally {
            clearInterval(timerInterval);
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    });

    // Close Citation Drawer
    closeDrawerBtn.addEventListener("click", closeDrawer);
    citationOverlay.addEventListener("click", closeDrawer);

    // ----------------------------------------------------
    // API Calls & Logic
    // ----------------------------------------------------
    
    function updateSessionMonogram() {
        if (sessionHashEl) {
            sessionHashEl.textContent = "HSX-" + sessionId.split("-")[0].toUpperCase();
        }
    }

    async function fetchSystemStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();
            
            modeText.textContent = data.mode.toUpperCase();
            modelText.innerHTML = `LLM: <code>${data.config.llm_model}</code><br>Embeddings: <code>${data.config.embeddings}</code>`;
        } catch (err) {
            console.error("Error fetching system status:", err);
            modeText.textContent = "SYSTEM OFFLINE";
        }
    }

    function updatePlaceholder(documents) {
        const userInput = document.getElementById("user-input");
        if (!userInput) return;
        
        if (documents && documents.length > 0) {
            // Find the latest document (last one in list)
            const latestDoc = documents[documents.length - 1].filename;
            let suggestedQuery = `What are the main findings in ${latestDoc}?`;
            
            const fnLower = latestDoc.toLowerCase();
            if (fnLower.includes("fairytale") || fnLower.includes("heroes") || fnLower.includes("aldric")) {
                suggestedQuery = "Who is King Aldric?";
            } else if (fnLower.includes("kant") || fnLower.includes("critique") || fnLower.includes("arendt")) {
                suggestedQuery = "What is Kant's concept of duty?";
            } else if (fnLower.includes("rich") || fnLower.includes("poor") || fnLower.includes("karl") || fnLower.includes("marx")) {
                suggestedQuery = "What is the critique of capitalism?";
            } else {
                // Strip extension for cleaner look
                const cleanName = latestDoc.replace(/\.[^/.]+$/, "");
                suggestedQuery = `What is the core topic of ${cleanName}?`;
            }
            userInput.placeholder = `Submit a query to the corpus (e.g. '${suggestedQuery}')...`;
        } else {
            userInput.placeholder = "Submit a query to the corpus (please upload a document first!)...";
        }
    }

    async function fetchDocuments() {
        docList.innerHTML = "<li class='loading-item'>Loading catalog...</li>";
        try {
            const res = await fetch("/api/documents");
            const data = await res.json();
            
            docList.innerHTML = "";
            if (!data.documents || data.documents.length === 0) {
                docList.innerHTML = "<li class='catalog-empty'>No texts indexed.</li>";
                updatePlaceholder(null);
                return;
            }

            updatePlaceholder(data.documents);

            data.documents.forEach(doc => {
                const li = document.createElement("li");
                li.className = "doc-item";
                li.innerHTML = `
                    <div style="display: flex; align-items: center; gap: 8px; width: 100%;">
                        <i class="fa-regular fa-file-pdf"></i>
                        <span class="doc-name" title="${doc.filename}" style="flex-grow: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${doc.filename}</span>
                        <button class="delete-doc-btn" data-filename="${doc.filename}" title="Delete file from database" style="background: none; border: none; color: #ff4a4a; cursor: pointer; padding: 0 5px;">
                            <i class="fa-solid fa-trash-can"></i>
                        </button>
                    </div>
                `;
                
                // Add delete listener
                const delBtn = li.querySelector(".delete-doc-btn");
                delBtn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    if (confirm(`Are you sure you want to delete '${doc.filename}' from the database?`)) {
                        delBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i>';
                        try {
                            const res = await fetch(`/api/documents/${encodeURIComponent(doc.filename)}`, { method: "DELETE" });
                            if (res.ok) {
                                fetchDocuments();
                            } else {
                                const data = await res.json();
                                alert("Failed to delete: " + data.message);
                                delBtn.innerHTML = '<i class="fa-solid fa-trash-can"></i>';
                            }
                        } catch (err) {
                            alert("Error deleting document: " + err.message);
                            delBtn.innerHTML = '<i class="fa-solid fa-trash-can"></i>';
                        }
                    }
                });

                docList.appendChild(li);
            });
        } catch (err) {
            console.error("Error loading documents:", err);
            docList.innerHTML = "<li class='catalog-empty'>Failed to load index.</li>";
            updatePlaceholder(null);
        }
    }

    // Handles the async uploading of files to the backend
    async function handleFileUpload(file) {
        const allowedExtensions = /(\.pdf|\.txt|\.md)$/i;
        if (!allowedExtensions.exec(file.name)) {
            setDropzoneStatus("error", "UNSUPPORTED TYPE", "PDF, TXT or MD only");
            setTimeout(resetDropzone, 3000);
            return;
        }

        const iconEl    = dropzone.querySelector(".dropzone-icon");
        const textEl    = dropzone.querySelector(".dropzone-text");
        const subtextEl = dropzone.querySelector(".dropzone-subtext");

        // Capture original state
        const origIcon    = iconEl.className;
        const origText    = textEl.textContent;
        const origSubtext = subtextEl.textContent;

        function resetDropzone() {
            iconEl.className  = origIcon;
            textEl.textContent    = origText;
            subtextEl.textContent = origSubtext;
            dropzone.classList.remove("dropzone-success", "dropzone-error");
            dropzone.style.pointerEvents = "auto";
            fileUploader.value = "";
        }

        function setDropzoneStatus(state, text, subtext, icon = null) {
            iconEl.className = (icon || (state === "error" ? "fa-solid fa-triangle-exclamation" : "fa-solid fa-circle-check")) + " dropzone-icon";
            textEl.textContent    = text;
            subtextEl.textContent = subtext;
            dropzone.classList.toggle("dropzone-success", state === "success");
            dropzone.classList.toggle("dropzone-error",   state === "error");
        }

        // Loading state
        iconEl.className      = "fa-solid fa-spinner fa-spin dropzone-icon";
        textEl.textContent    = "INDEXING SCROLL...";
        subtextEl.textContent = file.name;
        dropzone.style.pointerEvents = "none";

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });
            const data = await response.json();

            if (data.status === "success") {
                setDropzoneStatus("success", "INDEXED!", `${data.chunks} vectors · ${file.name}`);
                fetchDocuments();
                setTimeout(resetDropzone, 4000);
            } else {
                // Show the server error message inline
                const msg = data.message || "Unknown error";
                setDropzoneStatus("error", "UPLOAD FAILED", msg.length > 60 ? msg.slice(0, 57) + "…" : msg);
                console.error("Upload failed:", msg);
                setTimeout(resetDropzone, 6000);
            }
        } catch (err) {
            console.error("Upload network error:", err);
            setDropzoneStatus("error", "NETWORK ERROR", "Check console · may be a large doc (>40 pages)");
            setTimeout(resetDropzone, 5000);
        }
    }

    // Streams RAG response using ReadableStream API
    async function executeStreamingQuery(message, timerInterval) {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, session_id: sessionId })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        
        let entryDiv = null;
        let msgBody = null;
        let citationsBar = null;
        let citationsChips = null;
        let metricsBar = null;

        function ensureEntryCreated() {
            if (entryDiv) return;
            entryDiv = document.createElement("div");
            entryDiv.className = "manuscript-entry assistant-entry";
            entryDiv.innerHTML = `
                <div class="manuscript-body">
                    <div class="msg-body"></div>
                    <div class="codex-citations-bar hidden">
                        <span class="citations-label">References:</span>
                        <div class="citations-chips"></div>
                    </div>
                    <div class="codex-metrics-bar hidden"></div>
                </div>
            `;
            chatMessages.appendChild(entryDiv);
            msgBody = entryDiv.querySelector(".msg-body");
            citationsBar = entryDiv.querySelector(".codex-citations-bar");
            citationsChips = entryDiv.querySelector(".citations-chips");
            metricsBar = entryDiv.querySelector(".codex-metrics-bar");
            scrollToBottom();
        }

        const thinkingTxt = typingIndicator.querySelector(".thinking-txt");

        let responseText = "";
        let sources = [];
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split("\n\n");
            buffer = parts.pop(); // Keep partial line in buffer

            for (const part of parts) {
                if (part.trim() === "") continue;

                const eventMatch = part.match(/^event:\s*(\w+)/m);
                const dataMatch = part.match(/^data:\s*(.*)/m);

                if (eventMatch && dataMatch) {
                    const event = eventMatch[1];
                    const rawData = dataMatch[1];
                    
                    try {
                        const data = JSON.parse(rawData);
                        
                        if (event === "sources") {
                            sources = data;
                            ensureEntryCreated();
                        } else if (event === "token") {
                            ensureEntryCreated();
                            typingIndicator.classList.add("hidden");
                            responseText += data;
                            msgBody.innerHTML = formatMarkdown(responseText);
                            scrollToBottom();
                        } else if (event === "status") {
                            const elapsed = typingIndicator.querySelector(".stopwatch")?.textContent || "0.0s";
                            thinkingTxt.innerHTML = `${data} <span class='stopwatch'>${elapsed}</span>`;
                        } else if (event === "metrics") {
                            clearInterval(timerInterval);
                            ensureEntryCreated();
                            typingIndicator.classList.add("hidden");
                            metricsBar.innerHTML = `
                                <div class="metrics-item" title="Pipeline Latency"><i class="fa-solid fa-stopwatch"></i> ${data.latency}s</div>
                                <div class="metrics-item" title="Vector Chunks Retrieved"><i class="fa-solid fa-book-open"></i> ${data.chunks} chunks</div>
                                <div class="metrics-item" title="Graph DB Traversal"><i class="fa-solid fa-diagram-project"></i> ${typeof data.graph === 'number' ? `${data.graph} facts` : (data.graph ? "Traversed" : "Skipped")}</div>
                            `;
                            metricsBar.classList.remove("hidden");
                            scrollToBottom();
                        } else if (event === "done") {
                            clearInterval(timerInterval);
                            typingIndicator.classList.add("hidden");
                            break;
                        }
                    } catch (err) {
                        console.error("JSON parsing error:", part, err);
                    }
                }
            }
        }

        // Render Citations
        if (sources && sources.length > 0) {
            ensureEntryCreated();
            citationsBar.classList.remove("hidden");
            citationsChips.innerHTML = "";
            
            sources.forEach((source) => {
                const chip = document.createElement("button");
                chip.className = "footnote-pill";
                chip.innerHTML = `<i class="fa-solid fa-scroll"></i> P.${source.page} [${Math.round(source.score * 100)}%]`;
                
                chip.addEventListener("click", () => {
                    openDrawer(source);
                });
                
                citationsChips.appendChild(chip);
            });
        }
    }

    // ----------------------------------------------------
    // Drawer Management
    // ----------------------------------------------------
    
    function openDrawer(source) {
        citationSourceDoc.textContent = source.source;
        citationPage.textContent = `PAGE ${source.page}`;
        citationScore.textContent = `${(source.score * 100).toFixed(1)}% RELEVANCE`;
        
        // Verbatim quote
        citationText.textContent = source.snippet;
        
        citationOverlay.classList.add("active");
        citationDrawer.classList.add("active");
    }

    function closeDrawer() {
        citationOverlay.classList.remove("active");
        citationDrawer.classList.remove("active");
    }

    // ----------------------------------------------------
    // Helpers & Formatters
    // ----------------------------------------------------
    
    function appendMessage(role, text) {
        const entryDiv = document.createElement("div");
        entryDiv.className = `manuscript-entry ${role}-entry`;
        
        if (role === "user") {
            entryDiv.innerHTML = `<div class="query-title">${text}</div>`;
        } else {
            entryDiv.innerHTML = `
                <div class="manuscript-body">
                    <p>${text}</p>
                </div>
            `;
        }
        
        chatMessages.appendChild(entryDiv);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function generateUUID() {
        return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
            (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
        );
    }

    // Elegant Markdown parser utilizing marked.js
    function formatMarkdown(text) {
        if (!text) return "";
        // Strip carriage returns first
        const cleanText = text.replace(/\r/g, "");
        // Configure marked options
        marked.setOptions({
            gfm: true,
            breaks: true
        });
        return marked.parse(cleanText);
    }
});
