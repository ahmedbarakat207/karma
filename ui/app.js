(function() {
  'use strict';

  let ws = null;
  let wsConnected = false;
  let reconnectTimer = null;

  let currentState = {
    mood: 'playful',
    emotion: 'playful',
    energy: 0.85,
    curiosity: 0.70,
    speaking: false,
    gazeX: 0.5,
    gazeY: 0.5,
    activeCode: null,
    activeLang: 'c',
    kioskView: 'face',
    floorIdx: 0,
    speechText: ''
  };

  let kioskData = {
    studentApps: [
      {
        name: "Campus Wayfinder",
        author: "Omar & Sarah (Robotics Club)",
        category: "Navigation",
        description: "Autonomous topological pathfinding and indoor lab navigation for visitors.",
        version: "v1.2.0",
        status: "ONLINE",
        badge: "ROS 2"
      },
      {
        name: "Lab Occupancy Monitor",
        author: "Youssef K. (Computer Science)",
        category: "Vision & Edge",
        description: "Real-time edge-calibrated seat density and lab workstation availability.",
        version: "v2.0.1",
        status: "ACTIVE",
        badge: "YOLOv8"
      },
      {
        name: "AI Engineering Companion",
        author: "Nour & Tarek (AI Lab)",
        category: "Cognition",
        description: "Interactive oral engineering quiz and problem-solving assistant powered by Qwen.",
        version: "v1.0.0",
        status: "ONLINE",
        badge: "PyTorch"
      },
      {
        name: "12V Environmental Telemetry",
        author: "Mostafa A. (Mechatronics)",
        category: "Sensors & Power",
        description: "Live telemetry tracking AGM battery thermals, motor rail draw, and air quality.",
        version: "v3.1.0",
        status: "LOGGING",
        badge: "FastAPI"
      }
    ],
    achievements: [
      {
        title: "Autonomous Awakening",
        description: "Cold boots from 12V 9Ah AGM battery into full companion OS in <6s.",
        badge: "REV-01",
        date: "2026-09-02",
        status: "VERIFIED"
      },
      {
        title: "Edge Qwen Cognition",
        description: "Offline Qwen 2.5 0.5B Instruct neural cognition with 4096-token context window.",
        badge: "REV-02",
        date: "2026-09-02",
        status: "ACTIVE"
      },
      {
        title: "MarkItDown Vector RAG",
        description: "PDF ingestion with Microsoft MarkItDown and persistent sqlite-vec memory.",
        badge: "REV-03",
        date: "2026-09-03",
        status: "INDEXED"
      },
      {
        title: "Cortex-A72 Turbo",
        description: "Hardware overclock to 2.0 GHz (+33% generation speedup).",
        badge: "REV-04",
        date: "2026-09-02",
        status: "OVERCLOCKED"
      },
      {
        title: "135° Touch Interaction",
        description: "Mechatronic head tilts to 135 degrees for ergonomic standing touchscreen interaction.",
        badge: "REV-05",
        date: "2026-09-03",
        status: "CALIBRATED"
      },
      {
        title: "Sub-200ms Acoustic Loop",
        description: "Zero-latency real-time voice streaming with Faster-Whisper and Kokoro TTS.",
        badge: "REV-06",
        date: "2026-09-03",
        status: "STREAMING"
      }
    ],
    indexedDocs: [
      { source: "karma_manual.pdf" }
    ],
    docChunks: []
  };

  let mapZoom = 1.0;

  // DOM Elements
  const appContainer = document.getElementById('app-container');
  const systemTime = document.getElementById('system-time');
  const osStatusText = document.getElementById('os-status-text');
  const osPulseDot = document.querySelector('.os-pulse-dot');
  const hudBatteryVal = document.getElementById('hud-battery-val');
  const diagBattery = document.getElementById('diag-battery');
  const micStatusLabel = document.getElementById('mic-status-label');
  const menuBtn = document.getElementById('menu-btn');
  const readoutMood = document.getElementById('readout-mood');
  const faceLinkLabel = document.getElementById('face-link-label');
  const tickerLink = document.getElementById('ticker-link');

  const dockAudioBars = document.getElementById('dock-audio-bars');
  const dockTouchBtn = document.getElementById('dock-touch-btn');
  const dockFaceBtn = document.getElementById('dock-face-btn');

  const eyeLeftArc = document.getElementById('eye-left-arc');
  const eyeRightArc = document.getElementById('eye-right-arc');
  const eyeLeftContainer = document.getElementById('eye-left-container');
  const eyeRightContainer = document.getElementById('eye-right-container');
  const mouthPath = document.getElementById('mouth-path');

  const codeStage = document.getElementById('code-stage');
  const codeContent = document.getElementById('code-content');
  const codeLangBadge = document.getElementById('code-lang-badge');
  const codeFilename = document.getElementById('code-filename');
  const codeCopyBtn = document.getElementById('code-copy-btn');
  const codeCloseBtn = document.getElementById('code-close-btn');

  const subtitleBar = document.getElementById('subtitle-bar');
  const subtitleText = document.getElementById('subtitle-text');

  const kioskCloseBtn = document.getElementById('kiosk-close-btn');
  const tabButtons = document.querySelectorAll('.nav-tab-btn');
  const consolePanes = document.querySelectorAll('.console-pane');

  const cadLevelBtns = document.querySelectorAll('.cad-level-btn');
  const cadSvg = document.getElementById('cad-svg');
  const floorLayer0 = document.getElementById('floor-layer-0');
  const floorLayer1 = document.getElementById('floor-layer-1');
  const cadCoords = document.getElementById('cad-coords');
  const cadRooms = document.querySelectorAll('.cad-room');
  const roomPopover = document.getElementById('room-detail-popover');
  const popoverCode = document.getElementById('popover-code');
  const popoverStatus = document.getElementById('popover-status');
  const popoverName = document.getElementById('popover-name');
  const popoverDesc = document.getElementById('popover-desc');
  const popoverCap = document.getElementById('popover-cap');

  const mapZoomInBtn = document.getElementById('map-zoom-in');
  const mapZoomOutBtn = document.getElementById('map-zoom-out');
  const mapResetBtn = document.getElementById('map-reset');

  const studentAppsGrid = document.getElementById('student-apps-grid');
  const achievementsGrid = document.getElementById('achievements-grid');
  const docsList = document.getElementById('docs-list');
  const docsBody = document.getElementById('docs-body');
  const docsTitle = document.getElementById('docs-current-title');

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  // Clock Update
  function updateClock() {
    if (!systemTime) return;
    const now = new Date();
    const hrs = String(now.getHours()).padStart(2, '0');
    const mins = String(now.getMinutes()).padStart(2, '0');
    systemTime.textContent = `${hrs}:${mins}`;
  }
  updateClock();
  setInterval(updateClock, 10000);

  function setLinkState(online) {
    if (osPulseDot) osPulseDot.classList.toggle('live', online);
    if (faceLinkLabel) faceLinkLabel.textContent = online ? 'online' : 'offline';
    if (tickerLink) tickerLink.textContent = online ? 'online' : 'offline';
  }

  // WebSocket Client — protocol unchanged (server.py)
  function initWebSocket() {
    const wsUrl = 'ws://127.0.0.1:8765';
    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        wsConnected = true;
        if (osStatusText) osStatusText.textContent = currentState.kioskView === 'face' ? 'STANDBY' : 'CONSOLE';
        setLinkState(true);
        sendAction('get_initial_data');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleServerMessage(data);
        } catch (e) {
          console.error('Error parsing WS message:', e);
        }
      };

      ws.onclose = () => {
        wsConnected = false;
        if (osStatusText) osStatusText.textContent = 'OFFLINE';
        setLinkState(false);
        scheduleReconnect();
      };

      ws.onerror = () => {
        wsConnected = false;
      };
    } catch (e) {
      scheduleReconnect();
    }
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      initWebSocket();
    }, 1500);
  }

  function sendAction(action, payload = {}) {
    if (!wsConnected || !ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(JSON.stringify({ action, ...payload }));
    } catch (err) {
      console.warn('WS send failed:', err);
    }
  }

  function handleServerMessage(msg) {
    if (!msg || !msg.type) return;

    if (msg.type === 'state_update') {
      if (msg.mood) setMood(msg.mood);
      if (msg.emotion) setEmotion(msg.emotion);
      if (msg.energy !== undefined) setEnergy(msg.energy);
      if (msg.curiosity !== undefined) setCuriosity(msg.curiosity);
      if (msg.speaking !== undefined) setSpeaking(msg.speaking, msg.speech);
      if (msg.gaze_x !== undefined && msg.gaze_y !== undefined) {
        setGaze(msg.gaze_x, msg.gaze_y);
      }
      if (msg.code !== undefined) {
        setCodeDisplay(msg.code, msg.code_lang || 'c');
      }
      if (msg.kiosk_view && msg.kiosk_view !== currentState.kioskView) {
        if (msg.kiosk_view === 'face') {
          closeKiosk(false);
        } else {
          openKiosk(msg.kiosk_view, false);
        }
      }
    } else if (msg.type === 'kiosk_data') {
      if (msg.student_apps) {
        kioskData.studentApps = msg.student_apps;
        renderStudentApps();
      }
      if (msg.achievements) {
        kioskData.achievements = msg.achievements;
        renderAchievements();
      }
      if (msg.docs) {
        kioskData.indexedDocs = msg.docs;
        renderDocsSidebar();
      }
    } else if (msg.type === 'doc_chunks') {
      if (msg.source && docsTitle) docsTitle.textContent = msg.source;
      if (Array.isArray(msg.chunks) && docsBody) {
        docsBody.textContent = msg.chunks.length ? msg.chunks[0] : 'No passages in this document yet.';
      }
    }
  }

  function setMood(mood) {
    if (!mood) return;
    currentState.mood = mood.toLowerCase();
    if (readoutMood) readoutMood.textContent = currentState.mood.toUpperCase();
  }

  // Plain solid-circle eyes. Emotion only nudges the size.
  const EYE_FILL = '#4da3ff';
  const EYE_EXPRESSIONS = {
    playful:   "M -32 0 A 32 32 0 1 0 32 0 A 32 32 0 1 0 -32 0",
    curious:   "M -28 0 A 28 28 0 1 0 28 0 A 28 28 0 1 0 -28 0",
    warm:      "M -32 0 A 32 32 0 1 0 32 0 A 32 32 0 1 0 -32 0",
    excited:   "M -36 0 A 36 36 0 1 0 36 0 A 36 36 0 1 0 -36 0",
    tired:     "M -30 0 A 30 30 0 1 0 30 0 A 30 30 0 1 0 -30 0",
    attentive: "M -30 0 A 30 30 0 1 0 30 0 A 30 30 0 1 0 -30 0",
    neutral:   "M -31 0 A 31 31 0 1 0 31 0 A 31 31 0 1 0 -31 0"
  };

  const MOUTH_EXPRESSIONS = {
    playful:   "M -28 0 Q 0 26 28 0",
    curious:   "M -20 0 Q 0 14 20 0",
    warm:      "M -32 0 Q 0 32 32 0",
    excited:   "M -34 0 Q 0 38 34 0",
    tired:     "M -20 4 Q 0 10 20 4",
    attentive: "M -25 0 Q 0 22 25 0",
    neutral:   "M -25 0 Q 0 18 25 0"
  };

  function applyEye(d) {
    for (const el of [eyeLeftArc, eyeRightArc]) {
      el.setAttribute('d', d);
      el.setAttribute('fill', EYE_FILL);
      el.setAttribute('stroke-width', 2);
    }
  }

  function setEmotion(emotion) {
    currentState.emotion = (emotion || 'playful').toLowerCase();
    if (readoutMood) readoutMood.textContent = currentState.emotion.toUpperCase();
    applyEye(EYE_EXPRESSIONS[currentState.emotion] || EYE_EXPRESSIONS.playful);

    if (!currentState.speaking) {
      const mouthD = MOUTH_EXPRESSIONS[currentState.emotion] || MOUTH_EXPRESSIONS.playful;
      mouthPath.setAttribute('d', mouthD);
    }
  }

  function setEnergy(val) {
    currentState.energy = Math.max(0, Math.min(1, val));
    const pct = Math.round(currentState.energy * 100);
    const volt = (11.6 + currentState.energy * 0.9).toFixed(1);
    if (hudBatteryVal) hudBatteryVal.textContent = `${pct}%`;
    if (diagBattery) diagBattery.textContent = `${pct}% (${volt}V AGM)`;
  }

  function setCuriosity(val) {
    currentState.curiosity = Math.max(0, Math.min(1, val));
  }

  let speakInterval = null;
  function setSpeaking(isSpeaking, speech = '') {
    currentState.speaking = isSpeaking;
    currentState.speechText = speech;

    if (isSpeaking) {
      if (speech && speech.trim()) {
        subtitleText.textContent = speech.trim();
        subtitleBar.style.display = 'flex';
      }
      if (osStatusText) osStatusText.textContent = 'SPEAKING';
      if (micStatusLabel) micStatusLabel.textContent = 'TALKING';
      if (dockAudioBars) dockAudioBars.classList.add('speaking-active');

      if (!speakInterval) {
        let frame = 0;
        speakInterval = setInterval(() => {
          frame++;
          // small smile opens and closes as it talks
          const open = 8 + Math.round(Math.abs(Math.sin(frame * 0.9)) * 22);
          mouthPath.setAttribute('d', `M -28 0 Q 0 ${open} 28 0`);
        }, 90);
      }
    } else {
      if (subtitleBar) subtitleBar.style.display = 'none';
      if (osStatusText) {
        osStatusText.textContent = currentState.kioskView === 'face' ? 'STANDBY' : 'CONSOLE';
      }
      if (micStatusLabel) micStatusLabel.textContent = 'READY';
      if (dockAudioBars) dockAudioBars.classList.remove('speaking-active');

      if (speakInterval) {
        clearInterval(speakInterval);
        speakInterval = null;
      }
      const mouthD = MOUTH_EXPRESSIONS[currentState.emotion] || MOUTH_EXPRESSIONS.playful;
      mouthPath.setAttribute('d', mouthD);
    }
  }

  // Smooth Eye Gaze & Specular Tracking
  let targetGazeX = 0.5;
  let targetGazeY = 0.5;
  let currentGazeX = 0.5;
  let currentGazeY = 0.5;

  function setGaze(x, y) {
    targetGazeX = Math.max(0.1, Math.min(0.9, x));
    targetGazeY = Math.max(0.1, Math.min(0.9, y));
  }

  function updateGazeLoop() {
    currentGazeX += (targetGazeX - currentGazeX) * 0.12;
    currentGazeY += (targetGazeY - currentGazeY) * 0.12;

    const offsetX = (currentGazeX - 0.5) * 30;
    const offsetY = (currentGazeY - 0.5) * 16;

    eyeLeftContainer.setAttribute('transform', `translate(${280 + offsetX}, ${150 + offsetY})`);
    eyeRightContainer.setAttribute('transform', `translate(${520 + offsetX}, ${150 + offsetY})`);

    requestAnimationFrame(updateGazeLoop);
  }
  requestAnimationFrame(updateGazeLoop);

  // Plain blink: discs flatten to a line for an instant, then pop back
  let isBlinking = false;
  function triggerBlink() {
    if (isBlinking) return;
    isBlinking = true;

    const d = EYE_EXPRESSIONS[currentState.emotion] || EYE_EXPRESSIONS.playful;
    for (const el of [eyeLeftArc, eyeRightArc]) {
      el.setAttribute('d', "M -26 0 L 26 0");
      el.setAttribute('fill', 'none');
      el.setAttribute('stroke-width', 8);
    }

    setTimeout(() => {
      applyEye(d);
      isBlinking = false;
    }, 130);
  }

  function scheduleNextBlink() {
    const delay = 2800 + Math.random() * 3200;
    setTimeout(() => {
      triggerBlink();
      scheduleNextBlink();
    }, delay);
  }
  scheduleNextBlink();

  // Code Mode Presentation
  function setCodeDisplay(code, lang = 'c') {
    if (code && code.trim()) {
      currentState.activeCode = code;
      currentState.activeLang = lang || 'c';

      codeLangBadge.textContent = (lang || 'code').toUpperCase();
      codeFilename.textContent = `snippet.${lang || 'txt'}`;
      codeContent.textContent = code.trim();

      codeStage.style.display = 'flex';
    } else {
      currentState.activeCode = null;
      codeStage.style.display = 'none';
    }
  }

  function dismissCode() {
    setCodeDisplay(null);
    sendAction('clear_code');
  }

  codeCloseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    dismissCode();
  });

  codeStage.addEventListener('click', (e) => {
    if (e.target === codeStage) dismissCode();
  });

  codeCopyBtn.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (currentState.activeCode) {
      try {
        await navigator.clipboard.writeText(currentState.activeCode);
        const prev = codeCopyBtn.textContent;
        codeCopyBtn.textContent = 'COPIED';
        setTimeout(() => { codeCopyBtn.textContent = prev; }, 1500);
      } catch (err) {
        console.error('Copy error:', err);
      }
    }
  });

  // Kiosk Console Operations
  function openKiosk(view = 'map', notifyBackend = true) {
    currentState.kioskView = view;
    appContainer.classList.remove('mode-face');
    appContainer.classList.add('mode-kiosk');
    if (osStatusText) osStatusText.textContent = 'CONSOLE';
    switchKioskTab(view);
    if (notifyBackend) {
      sendAction('open_view', { view });
    }
  }

  function closeKiosk(notifyBackend = true) {
    currentState.kioskView = 'face';
    appContainer.classList.remove('mode-kiosk');
    appContainer.classList.add('mode-face');
    if (osStatusText) osStatusText.textContent = wsConnected ? 'STANDBY' : 'OFFLINE';
    if (roomPopover) roomPopover.style.display = 'none';
    if (notifyBackend) {
      sendAction('close_kiosk');
    }
  }

  menuBtn.addEventListener('click', () => {
    openKiosk('map', true);
  });

  kioskCloseBtn.addEventListener('click', () => {
    closeKiosk(true);
  });

  // Head Tilt Quick Controls
  if (dockTouchBtn) {
    dockTouchBtn.addEventListener('click', () => {
      dockTouchBtn.classList.add('active');
      if (dockFaceBtn) dockFaceBtn.classList.remove('active');
      sendAction('tilt_touch');
    });
  }
  if (dockFaceBtn) {
    dockFaceBtn.addEventListener('click', () => {
      dockFaceBtn.classList.add('active');
      if (dockTouchBtn) dockTouchBtn.classList.remove('active');
      sendAction('tilt_face');
    });
  }

  // Diagnostic Actuator Action Buttons
  const actTiltTouch = document.getElementById('act-tilt-touch');
  const actTiltFace = document.getElementById('act-tilt-face');
  const actRestartUi = document.getElementById('act-restart-ui');
  if (actTiltTouch) actTiltTouch.addEventListener('click', () => sendAction('tilt_touch'));
  if (actTiltFace) actTiltFace.addEventListener('click', () => sendAction('tilt_face'));
  if (actRestartUi) actRestartUi.addEventListener('click', () => window.location.reload());

  // Navigation Tab Switching
  function switchKioskTab(viewName) {
    tabButtons.forEach(btn => {
      const on = btn.getAttribute('data-view') === viewName;
      btn.classList.toggle('active', on);
      btn.setAttribute('aria-selected', on ? 'true' : 'false');
    });

    consolePanes.forEach(pane => {
      pane.classList.toggle('active', pane.id === `view-${viewName}`);
    });

    if (viewName === 'apps' && kioskData.studentApps.length === 0) {
      sendAction('get_student_apps');
    } else if (viewName === 'achievements' && kioskData.achievements.length === 0) {
      sendAction('get_achievements');
    } else if (viewName === 'docs' && kioskData.indexedDocs.length === 0) {
      sendAction('get_docs');
    }
  }

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const view = btn.getAttribute('data-view');
      openKiosk(view, true);
    });
  });

  // CAD Blueprint Level Switching
  cadLevelBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      cadLevelBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const floor = parseInt(btn.getAttribute('data-floor'), 10);
      switchCadFloor(floor);
    });
  });

  function switchCadFloor(floor) {
    currentState.floorIdx = floor;
    if (floor === 1) {
      if (floorLayer0) floorLayer0.style.display = 'none';
      if (floorLayer1) floorLayer1.style.display = 'inline';
      if (cadCoords) cadCoords.textContent = 'X: 42.8\u2003\u2003Y: 34.0';
    } else {
      if (floorLayer0) floorLayer0.style.display = 'inline';
      if (floorLayer1) floorLayer1.style.display = 'none';
      if (cadCoords) cadCoords.textContent = 'X: 64.2\u2003\u2003Y: 58.1';
    }
    if (roomPopover) roomPopover.style.display = 'none';
    resetCadZoom();
    sendAction('switch_floor', { floor });
  }

  // Interactive CAD Room Inspection
  cadRooms.forEach(room => {
    room.addEventListener('click', (e) => {
      e.stopPropagation();
      const code = room.getAttribute('data-code');
      const name = room.getAttribute('data-room');
      const cap = room.getAttribute('data-capacity');
      const status = room.getAttribute('data-status');
      const desc = room.querySelector('.cad-room-desc')?.textContent || 'Research facility';

      if (popoverCode) popoverCode.textContent = code;
      if (popoverStatus) popoverStatus.textContent = (status || '').toUpperCase();
      if (popoverName) popoverName.textContent = name;
      if (popoverDesc) popoverDesc.textContent = desc;
      if (popoverCap) popoverCap.textContent = cap;

      if (roomPopover) roomPopover.style.display = 'block';
    });
  });

  if (roomPopover) {
    roomPopover.addEventListener('click', () => {
      roomPopover.style.display = 'none';
    });
  }

  // CAD Zoom Controls
  function resetCadZoom() {
    mapZoom = 1.0;
    updateCadTransform();
  }

  function updateCadTransform() {
    if (cadSvg) {
      cadSvg.style.transform = `scale(${mapZoom})`;
    }
  }

  if (mapZoomInBtn) {
    mapZoomInBtn.addEventListener('click', () => {
      mapZoom = Math.min(2.5, mapZoom + 0.25);
      updateCadTransform();
    });
  }

  if (mapZoomOutBtn) {
    mapZoomOutBtn.addEventListener('click', () => {
      mapZoom = Math.max(0.75, mapZoom - 0.25);
      updateCadTransform();
    });
  }

  if (mapResetBtn) {
    mapResetBtn.addEventListener('click', resetCadZoom);
  }

  function renderStudentApps() {
    if (!studentAppsGrid) return;
    studentAppsGrid.innerHTML = '';
    const apps = kioskData.studentApps || [];
    const appsCount = document.getElementById('apps-count');
    if (appsCount) appsCount.textContent = apps.length;
    apps.forEach((app, i) => {
      const row = document.createElement('div');
      row.className = 'index-row';
      const idx = String(i + 1).padStart(2, '0');
      row.innerHTML = `
        <span class="row-idx">${idx}</span>
        <span class="row-main">
          <span class="row-title">${esc(app.name)}</span>
          <span class="row-sub"><b>${esc(app.author || 'Autonomous Project')}</b> · ${esc(app.category || 'Project')} · ${esc(app.version || 'v1.0.0')}</span>
          <span class="row-desc">${esc(app.description || '')}</span>
        </span>
        <span class="row-side">
          <span class="row-flag">${esc(app.badge || app.category || 'APP')}</span>
          <span class="row-stat">${esc(app.status || 'DEPLOYED')}</span>
        </span>
      `;
      studentAppsGrid.appendChild(row);
    });
  }

  function renderAchievements() {
    if (!achievementsGrid) return;
    achievementsGrid.innerHTML = '';
    const list = kioskData.achievements || [];
    list.forEach((item, i) => {
      const row = document.createElement('div');
      row.className = 'index-row';
      const idx = String(i + 1).padStart(2, '0');
      row.innerHTML = `
        <span class="row-idx">${idx}</span>
        <span class="row-main">
          <span class="row-title">${esc(item.title)}</span>
          <span class="row-sub"><b>${esc(item.date || '2026')}</b> · ENGINEERING BADGE</span>
          <span class="row-desc">${esc(item.description)}</span>
        </span>
        <span class="row-side">
          <span class="row-flag solid">${esc(item.badge || 'BADGE')}</span>
          <span class="row-stat">${esc(item.status || 'VERIFIED')}</span>
        </span>
      `;
      achievementsGrid.appendChild(row);
    });
  }

  // Render Documentation Browser
  function renderDocsSidebar() {
    if (!docsList) return;
    docsList.innerHTML = '';
    const docs = kioskData.indexedDocs || [];

    const head = document.createElement('div');
    head.className = 'docs-side-head';
    head.textContent = `INDEX ▸ ${docs.length} FILE${docs.length === 1 ? '' : 'S'}`;
    docsList.appendChild(head);

    docs.forEach((doc, idx) => {
      const btn = document.createElement('button');
      btn.className = `doc-nav-btn ${idx === 0 ? 'active' : ''}`;
      btn.textContent = doc.source || `Document ${idx + 1}`;
      btn.addEventListener('click', () => {
        document.querySelectorAll('.doc-nav-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        if (docsTitle) docsTitle.textContent = doc.source;
        sendAction('get_doc_chunks', { source: doc.source });
      });
      docsList.appendChild(btn);
    });

    if (docs.length > 0) {
      if (docsTitle) docsTitle.textContent = docs[0].source;
      if (docsBody && !docsBody.textContent.trim()) {
        docsBody.textContent = [
          'KARMA AUTONOMOUS COMPANION — SYSTEM SPEC',
          '──────────────────────────────────────',
          'HOST ...... Raspberry Pi 4B / Cortex-A72 @ 2.0GHz',
          'NEURAL .... Qwen 2.5 0.5B Instruct / GGUF / 4096 CTX',
          'NECK ...... 90° gaze / 135° touch pitch servo',
          'POWER ..... 12V 9.0Ah AGM / dual BTS7960 43A',
          'VISION .... 720p + face mesh + gaze estimation',
          'VOICE ..... Faster-Whisper ASR + Kokoro TTS',
          '',
          'Select a file on the left to page its passages.'
        ].join('\n');
      }
    } else if (docsBody) {
      docsBody.textContent = 'No documents indexed yet.';
    }
  }

  // Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (currentState.kioskView !== 'face') {
        closeKiosk(true);
      } else if (currentState.activeCode) {
        dismissCode();
      }
    } else if (e.key.toLowerCase() === 'm' && !e.metaKey && !e.ctrlKey) {
      if (currentState.kioskView === 'face') {
        openKiosk('map', true);
      } else {
        closeKiosk(true);
      }
    }
  });

  // Initialization
  setLinkState(false);
  setEmotion(currentState.emotion);
  renderStudentApps();
  renderAchievements();
  renderDocsSidebar();
  initWebSocket();
})();
