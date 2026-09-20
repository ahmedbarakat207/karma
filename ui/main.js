const { app, BrowserWindow, screen } = require('electron');
const path = require('path');

app.commandLine.appendSwitch('no-sandbox');
app.commandLine.appendSwitch('disable-gpu-sandbox');
// Prevent Chromium auto-scaling / overscroll from cropping the kiosk:
// force 1:1 device pixels (our own zoomFactor below handles fitting),
// disable pinch-zoom which can leave the console permanently cropped.
app.commandLine.appendSwitch('force-device-scale-factor', '1');
app.commandLine.appendSwitch('disable-pinch');
app.commandLine.appendSwitch('overscroll-history-navigation', '0');

let mainWindow = null;

// Design canvas the CSS is authored against. Anything smaller (e.g. an
// 800x480 panel) gets a zoomFactor < 1 so the whole console fits with
// nothing cut off; anything larger gets up to 1.25x so extra pixels are
// actually used ("increase the resolution") instead of letterboxed.
const DESIGN_W = parseInt(process.env.UI_WIDTH || '1024', 10) || 1024;
const DESIGN_H = parseInt(process.env.UI_HEIGHT || '600', 10) || 600;

function computeZoom(sw, sh) {
  // Explicit override: UI_ZOOM=0.85 or --zoom=0.85
  const flagArg = process.argv.find((a) => a.startsWith('--zoom='));
  const raw = flagArg ? flagArg.split('=')[1] : process.env.UI_ZOOM;
  const forced = raw != null && raw !== '' ? parseFloat(raw) : NaN;
  if (Number.isFinite(forced) && forced > 0) {
    return Math.min(2.0, Math.max(0.5, forced));
  }
  if (!sw || !sh) return 1.0;
  const fit = Math.min(sw / DESIGN_W, sh / DESIGN_H);
  // Never upscale tiny amounts (avoids blur); allow modest upscale on
  // large monitors so the UI genuinely gains resolution there.
  return Math.min(1.25, Math.max(0.6, fit));
}

function createWindow() {
  // Kiosk appliance: fullscreen by default. Opt out only with --windowed
  // (the old opt-in --kiosk flag is still honoured). The launcher
  // (src/main.py) always passes --kiosk on the robot.
  const windowed = process.argv.includes('--windowed');
  const isKiosk = !windowed;

  // Native panel on this unit is 1024x600 (was 800x480). Fit whatever
  // display is actually attached, but fall back to 1024x600 — the old
  // 800x480 default left console content cut off on the right/bottom.
  let sw = 1024;
  let sh = 600;
  try {
    const disp = screen.getPrimaryDisplay();
    if (disp && disp.workAreaSize) {
      sw = disp.workAreaSize.width || sw;
      sh = disp.workAreaSize.height || sh;
    }
  } catch (e) {
    console.error('[karma-ui] screen query failed, using 1024x600:', e);
  }
  const winW = windowed ? Math.min(1024, sw) : sw;
  const winH = windowed ? Math.min(600, sh) : sh;
  const zoom = computeZoom(sw, sh);
  console.error(`[karma-ui] creating window ${winW}x${winH} kiosk=${isKiosk} screen=${sw}x${sh} design=${DESIGN_W}x${DESIGN_H} zoom=${zoom.toFixed(3)}`);

  mainWindow = new BrowserWindow({
    width: winW,
    height: winH,
    minWidth: Math.min(400, winW),
    minHeight: Math.min(300, winH),
    useContentSize: true,
    backgroundColor: '#000000',
    frame: false,
    fullscreen: isKiosk,
    kiosk: isKiosk,
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      zoomFactor: zoom,
    }
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html')).catch((e) => {
    console.error('[karma-ui] loadFile failed:', e);
  });

  // Re-apply after load (zoomFactor in webPreferences can be dropped on
  // some Electron/Chromium versions) + keep it across navigations.
  try {
    if (mainWindow.webContents && typeof mainWindow.webContents.setZoomFactor === 'function') {
      mainWindow.webContents.setZoomFactor(zoom);
    }
  } catch (_) { /* ignore */ }
  mainWindow.webContents.on('did-finish-load', () => {
    try {
      mainWindow.webContents.setZoomFactor(zoom);
      // Expose the real viewport to the renderer so app.js can fine-tune
      // its own --ui-scale variable for sub-pixel fitting.
      mainWindow.webContents.executeJavaScript(
        `window.__KARMA_ZOOM=${zoom};window.dispatchEvent(new Event('karma-zoom'));`
      ).catch(() => {});
    } catch (_) { /* ignore */ }
  });

  // Show once the renderer is ready — but never gate visibility on it:
  // on Pi-class GPUs first paint can take very long (or never signal),
  // which used to leave the LCD on the black openbox background forever.
  let shown = false;
  const showNow = (why) => {
    try {
      if (isKiosk) {
        mainWindow.setKiosk(true);
        mainWindow.setFullScreen(true);
      } else {
        mainWindow.center();
      }
      mainWindow.show();
      mainWindow.focus();
      shown = true;
      console.error(`[karma-ui] shown (${why}) bounds=${JSON.stringify(mainWindow.getBounds())}`);
    } catch (e) {
      console.error('[karma-ui] show failed:', e);
      try { mainWindow.show(); shown = true; } catch (_) { /* ignore */ }
    }
  };

  mainWindow.once('ready-to-show', () => {
    console.error('[karma-ui] ready-to-show fired');
    showNow('ready-to-show');
  });

  // Safety net: show unconditionally. An isVisible() gate does NOT work
  // here — Electron reports kiosk windows visible while still unmapped.
  setTimeout(() => {
    try {
      if (mainWindow && !shown) {
        console.error('[karma-ui] ready-to-show timeout, forcing show');
        showNow('timeout');
      }
    } catch (_) { /* ignore */ }
  }, 3000);

  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
