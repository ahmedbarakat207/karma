(() => {
  let ws = null;
  let wsConnected = false;

  let currentState = {
    mood: 'playful',
    emotion: 'playful',
    energy: 0.85,
    curiosity: 0.70,
    speaking: false,
    speechText: '',
    gazeX: 0.5,
    gazeY: 0.5,
    activeCode: null,
    activeLang: 'c',
    kioskView: 'face',
    floorIdx: 0
  };

  let kioskData = {
    studentApps: [
      {
        name: "Campus Wayfinder",
        author: "Omar & Sarah (Robotics Club)",
        category: "Navigation",
        description: "Interactive floor-by-floor navigation and lab locator for students & visitors.",
        version: "1.2.0",
        status: "Active"
      },
      {
        name: "Lecture Hall Scheduler",
        author: "Youssef K. (Computer Science)",
        category: "Productivity",
        description: "Real-time room occupancy, lab availability, and lecture timetable viewer.",
        version: "2.0.1",
        status: "Active"
      },
      {
        name: "AI Study Buddy",
        author: "Nour & Tarek (AI Lab)",
        category: "Cognition",
        description: "Voice-interactive engineering quiz and flashcard companion powered by Qwen.",
        version: "1.0.0",
        status: "Active"
      },
      {
        name: "Lab Environmental Monitor",
        author: "Mostafa A. (Mechatronics)",
        category: "Sensors",
        description: "Live telemetry from temperature, air quality, and 12V 9Ah battery monitors.",
        version: "3.1.0",
        status: "Active"
      }
    ],
    achievements: [
      {
        title: "Autonomous Awakening",
        description: "Cold boots from 12V battery into fullscreen companion face in <6s.",
        badge: "⚡ POWER",
        date: "2026-09-02"
      },
      {
        title: "Pure Local Mind",
        description: "Offline Qwen 2.5 0.5B Instruct cognition with 4096-token context window.",
        badge: "🧠 BRAIN",
        date: "2026-09-02"
      },
      {
        title: "Document Scholar",
        description: "PDF ingestion with Microsoft MarkItDown and persistent sqlite-vec memory.",
        badge: "📚 RAG",
        date: "2026-09-03"
      },
      {
        title: "Cortex-A72 Turbo",
        description: "Hardware overclock to 2.0 GHz (+33% generation speedup).",
        badge: "🚀 OVERCLOCK",
        date: "2026-09-02"
      },
      {
        title: "135° Kiosk Pitch",
        description: "Head tilts to 135 degrees for natural standing touchscreen interaction.",
        badge: "🤖 TOUCH",
        date: "2026-09-03"
      },
      {
        title: "Fluent Acoustic Loop",
        description: "Zero-latency real-time voice streaming with Faster-Whisper and Kokoro TTS.",
        badge: "🗣️ AUDIO",
        date: "2026-09-03"
      }
    ],
    indexedDocs: [
      { source: "karma_manual.pdf" }
    ],
    docChunks: []
  };

  let mapZoom = 1.0;

  const appContainer = document.getElementById('app-container');
  const moodPill = document.getElementById('mood-pill');
  const moodLabel = document.getElementById('mood-label');
  const energyFill = document.getElementById('energy-fill');
  const curiosityFill = document.getElementById('curiosity-fill');
  const menuBtn = document.getElementById('menu-btn');

  const faceStage = document.getElementById('companion-station');
  const companionStatusNote = document.getElementById('companion-status-note');
  const dockTouchBtn = document.getElementById('dock-touch-btn');
  const dockFaceBtn = document.getElementById('dock-face-btn');

  const eyeLeftArc = document.getElementById('eye-left-arc');
  const eyeRightArc = document.getElementById('eye-right-arc');
  const eyeLeftPupil = document.getElementById('eye-left-pupil');
  const eyeRightPupil = document.getElementById('eye-right-pupil');
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

  const kioskOverlay = document.getElementById('kiosk-overlay');
  const kioskBackdrop = document.getElementById('kiosk-backdrop');
  const kioskCloseBtn = document.getElementById('kiosk-close-btn');
  const tabButtons = document.querySelectorAll('.tab-btn');
  const kioskViews = document.querySelectorAll('.kiosk-pane');

  const floorPillBtns = document.querySelectorAll('.floor-pill-btn');
  const mapImage = document.getElementById('map-image');
  const karmaBeacon = document.getElementById('karma-beacon');
  const mapZoomInBtn = document.getElementById('map-zoom-in');
  const mapZoomOutBtn = document.getElementById('map-zoom-out');
  const mapResetBtn = document.getElementById('map-reset');

  const studentAppsGrid = document.getElementById('student-apps-grid');
  const achievementsGrid = document.getElementById('achievements-grid');
  const docsList = document.getElementById('docs-list');
  const docsBody = document.getElementById('docs-body');
  const docsTitle = document.getElementById('docs-current-title');

  function initWebSocket() {
    const wsUrl = 'ws://127.0.0.1:8765';
    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        wsConnected = true;
        sendAction('get_initial_data');
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          handleServerMessage(msg);
        } catch (err) {
          console.error('[ui] ws parse error:', err);
        }
      };

      ws.onclose = () => {
        wsConnected = false;
        setTimeout(initWebSocket, 1500);
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch (e) {
      setTimeout(initWebSocket, 2000);
    }
  }

  function sendAction(action, payload = {}) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action, ...payload }));
    }
  }

  function handleServerMessage(msg) {
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
        setCodeDisplay(msg.code, msg.code_lang);
      }
      if (msg.kiosk_view !== undefined) {
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
      kioskData.docChunks = msg.chunks || [];
      if (docsBody) {
        docsBody.textContent = kioskData.docChunks.join('\n\n---\n\n') || 'No content found.';
      }
    }
  }

  function setMood(mood) {
    currentState.mood = mood.toLowerCase();
    moodLabel.textContent = mood.toUpperCase();
    moodPill.className = `pill mood-${currentState.mood}`;
  }

  const EYE_EXPRESSIONS = {
    playful: "M -55 20 A 55 55 0 0 1 55 20",
    curious: "M -48 10 A 48 48 0 0 1 48 10",
    warm: "M -50 16 A 50 50 0 0 1 50 16",
    excited: "M -58 24 A 58 58 0 0 1 58 24",
    tired: "M -45 -10 A 45 45 0 0 0 45 -10",
    attentive: "M -52 14 A 52 52 0 0 1 52 14",
    neutral: "M -50 0 L 50 0"
  };

  const MOUTH_EXPRESSIONS = {
    playful: "M -75 -5 Q 0 38 75 -5",
    curious: "M -40 0 Q 0 25 40 0",
    warm: "M -65 -4 Q 0 28 65 -4",
    excited: "M -80 -8 Q 0 48 80 -8",
    tired: "M -50 12 Q 0 -5 50 12",
    attentive: "M -55 -2 Q 0 18 55 -2",
    neutral: "M -50 0 L 50 0"
  };

  function setEmotion(emotion) {
    currentState.emotion = emotion.toLowerCase();
    const eyeD = EYE_EXPRESSIONS[currentState.emotion] || EYE_EXPRESSIONS.playful;
    eyeLeftArc.setAttribute('d', eyeD);
    eyeRightArc.setAttribute('d', eyeD);

    if (!currentState.speaking) {
      const mouthD = MOUTH_EXPRESSIONS[currentState.emotion] || MOUTH_EXPRESSIONS.playful;
      mouthPath.setAttribute('d', mouthD);
    }
  }

  function setEnergy(val) {
    currentState.energy = Math.max(0, Math.min(1, val));
    energyFill.style.width = `${Math.round(currentState.energy * 100)}%`;
  }

  function setCuriosity(val) {
    currentState.curiosity = Math.max(0, Math.min(1, val));
    curiosityFill.style.width = `${Math.round(currentState.curiosity * 100)}%`;
  }

  let speakInterval = null;
  function setSpeaking(isSpeaking, speech = '') {
    currentState.speaking = isSpeaking;
    currentState.speechText = speech;

    if (isSpeaking) {
      if (speech && speech.trim()) {
        subtitleText.textContent = speech.trim();
        subtitleBar.style.display = 'block';
      }
      if (!speakInterval) {
        let toggle = false;
        speakInterval = setInterval(() => {
          toggle = !toggle;
          const openH = toggle ? 35 : 12;
          mouthPath.setAttribute('d', `M -65 0 Q 0 ${openH} 65 0`);
        }, 120);
      }
    } else {
      if (speakInterval) {
        clearInterval(speakInterval);
        speakInterval = null;
      }
      const mouthD = MOUTH_EXPRESSIONS[currentState.emotion] || MOUTH_EXPRESSIONS.playful;
      mouthPath.setAttribute('d', mouthD);
      setTimeout(() => {
        if (!currentState.speaking) {
          subtitleBar.style.display = 'none';
        }
      }, 3000);
    }
  }

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

    const offsetX = (currentGazeX - 0.5) * 35;
    const offsetY = (currentGazeY - 0.5) * 20;

    eyeLeftContainer.setAttribute('transform', `translate(${240 + offsetX}, ${160 + offsetY})`);
    eyeRightContainer.setAttribute('transform', `translate(${560 + offsetX}, ${160 + offsetY})`);

    requestAnimationFrame(updateGazeLoop);
  }
  requestAnimationFrame(updateGazeLoop);

  let isBlinking = false;
  function triggerBlink() {
    if (isBlinking || currentState.emotion === 'tired') return;
    isBlinking = true;

    const origEye = EYE_EXPRESSIONS[currentState.emotion] || EYE_EXPRESSIONS.playful;
    eyeLeftArc.setAttribute('d', "M -50 0 L 50 0");
    eyeRightArc.setAttribute('d', "M -50 0 L 50 0");

    setTimeout(() => {
      eyeLeftArc.setAttribute('d', origEye);
      eyeRightArc.setAttribute('d', origEye);
      isBlinking = false;
    }, 140);
  }

  function scheduleNextBlink() {
    const delay = 2500 + Math.random() * 3500;
    setTimeout(() => {
      triggerBlink();
      scheduleNextBlink();
    }, delay);
  }
  scheduleNextBlink();

  function setCodeDisplay(code, lang = 'c') {
    if (code && code.trim()) {
      currentState.activeCode = code;
      currentState.activeLang = lang || 'c';

      codeLangBadge.textContent = (lang || 'code').toUpperCase();
      codeFilename.textContent = `snippet.${lang || 'txt'}`;
      codeContent.textContent = code.trim();

      appContainer.classList.add('mode-coding');
      codeStage.style.display = 'flex';
    } else {
      currentState.activeCode = null;
      appContainer.classList.remove('mode-coding');
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
    if (e.target === codeStage) {
      dismissCode();
    }
  });

  codeCopyBtn.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (currentState.activeCode) {
      try {
        await navigator.clipboard.writeText(currentState.activeCode);
        const label = codeCopyBtn.querySelector('.btn-label');
        if (label) {
          label.textContent = 'Copied!';
          setTimeout(() => label.textContent = 'Copy', 1800);
        }
      } catch (err) {
        console.error('Clipboard copy error:', err);
      }
    }
  });

  function openKiosk(view = 'map', notifyBackend = true) {
    currentState.kioskView = view;
    appContainer.classList.remove('mode-face');
    appContainer.classList.add('mode-kiosk');
    if (companionStatusNote) {
      companionStatusNote.textContent = 'Browsing Interactive Hub';
    }
    switchKioskTab(view);
    if (notifyBackend) {
      sendAction('open_view', { view });
    }
  }

  function closeKiosk(notifyBackend = true) {
    currentState.kioskView = 'face';
    appContainer.classList.remove('mode-kiosk');
    appContainer.classList.add('mode-face');
    if (companionStatusNote) {
      companionStatusNote.textContent = 'Hanging out & listening to you';
    }
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

  if (dockTouchBtn) {
    dockTouchBtn.addEventListener('click', () => sendAction('tilt_touch'));
  }
  if (dockFaceBtn) {
    dockFaceBtn.addEventListener('click', () => sendAction('tilt_face'));
  }

  function switchKioskTab(viewName) {
    tabButtons.forEach(btn => {
      if (btn.getAttribute('data-view') === viewName) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    kioskViews.forEach(v => {
      if (v.id === `view-${viewName}`) {
        v.classList.add('active');
      } else {
        v.classList.remove('active');
      }
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
      switchKioskTab(view);
      sendAction('switch_view', { view });
    });
  });

  floorPillBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      floorPillBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const floorIdx = parseInt(btn.getAttribute('data-floor'), 10);
      switchFloor(floorIdx);
    });
  });

  function switchFloor(idx) {
    currentState.floorIdx = idx;
    mapImage.src = idx === 1 ? '../data/maps/floor_2.jpg' : '../data/maps/floor_1.jpg';
    if (karmaBeacon) {
      if (idx === 1) {
        karmaBeacon.style.top = '36%';
        karmaBeacon.style.left = '42%';
      } else {
        karmaBeacon.style.top = '61.8%';
        karmaBeacon.style.left = '62.5%';
      }
    }
    resetMapZoom();
    sendAction('switch_floor', { floor: idx });
  }

  function updateMapTransform() {
    mapImage.style.transform = `scale(${mapZoom})`;
  }

  function resetMapZoom() {
    mapZoom = 1.0;
    updateMapTransform();
  }

  mapZoomInBtn.addEventListener('click', () => {
    mapZoom = Math.min(2.5, mapZoom + 0.25);
    updateMapTransform();
  });

  mapZoomOutBtn.addEventListener('click', () => {
    mapZoom = Math.max(0.75, mapZoom - 0.25);
    updateMapTransform();
  });

  mapResetBtn.addEventListener('click', () => {
    resetMapZoom();
  });

  function renderStudentApps() {
    if (!studentAppsGrid) return;
    studentAppsGrid.innerHTML = '';
    kioskData.studentApps.forEach(app => {
      const card = document.createElement('div');
      card.className = 'showcase-card';
      card.innerHTML = `
        <div class="card-top">
          <div>
            <div class="card-title">${escapeHtml(app.name)}</div>
            <div class="card-author">by ${escapeHtml(app.author)}</div>
          </div>
          <span class="card-badge">${escapeHtml(app.category || 'App')}</span>
        </div>
        <div class="card-desc">${escapeHtml(app.description)}</div>
        <div class="card-meta">
          <span>v${escapeHtml(app.version || '1.0')}</span>
          <span style="color: var(--neon-cyan);">${escapeHtml(app.status || 'Active')}</span>
        </div>
      `;
      studentAppsGrid.appendChild(card);
    });
  }

  function renderAchievements() {
    if (!achievementsGrid) return;
    achievementsGrid.innerHTML = '';
    kioskData.achievements.forEach(ach => {
      const card = document.createElement('div');
      card.className = 'showcase-card';
      card.innerHTML = `
        <div class="card-top">
          <div class="card-title">${escapeHtml(ach.title)}</div>
          <span class="card-badge" style="background: rgba(255, 208, 0, 0.15); border-color: rgba(255, 208, 0, 0.4); color: var(--neon-gold);">${escapeHtml(ach.badge || '🏆')}</span>
        </div>
        <div class="card-desc">${escapeHtml(ach.description)}</div>
        <div class="card-meta">
          <span>Unlocked</span>
          <span>${escapeHtml(ach.date || '2026')}</span>
        </div>
      `;
      achievementsGrid.appendChild(card);
    });
  }

  function renderDocsSidebar() {
    if (!docsList) return;
    docsList.innerHTML = '';
    kioskData.indexedDocs.forEach((doc, idx) => {
      const btn = document.createElement('button');
      btn.className = `doc-item-btn ${idx === 0 ? 'active' : ''}`;
      btn.textContent = doc.source || `Document ${idx + 1}`;
      btn.title = doc.source;
      btn.addEventListener('click', () => {
        document.querySelectorAll('.doc-item-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        if (docsTitle) docsTitle.textContent = doc.source;
        sendAction('get_doc_chunks', { source: doc.source });
      });
      docsList.appendChild(btn);
    });

    if (kioskData.indexedDocs.length > 0) {
      const first = kioskData.indexedDocs[0];
      if (docsTitle) docsTitle.textContent = first.source;
      sendAction('get_doc_chunks', { source: first.source });
    }
  }

  const actTiltTouch = document.getElementById('act-tilt-touch');
  const actTiltFace = document.getElementById('act-tilt-face');
  const actRestartUi = document.getElementById('act-restart-ui');

  if (actTiltTouch) {
    actTiltTouch.addEventListener('click', () => sendAction('tilt_touch'));
  }
  if (actTiltFace) {
    actTiltFace.addEventListener('click', () => sendAction('tilt_face'));
  }
  if (actRestartUi) {
    actRestartUi.addEventListener('click', () => window.location.reload());
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (currentState.kioskView !== 'face') {
        closeKiosk(true);
      } else if (currentState.activeCode) {
        dismissCode();
      }
    } else if (e.key.toLowerCase() === 'm') {
      if (currentState.kioskView === 'face') {
        openKiosk('map', true);
      } else {
        closeKiosk(true);
      }
    }
  });

  renderStudentApps();
  renderAchievements();
  renderDocsSidebar();

  initWebSocket();
})();
