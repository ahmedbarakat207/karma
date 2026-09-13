const { app, BrowserWindow, screen } = require('electron');
const path = require('path');

app.commandLine.appendSwitch('no-sandbox');
app.commandLine.appendSwitch('disable-gpu-sandbox');

let mainWindow = null;

function createWindow() {
  // Kiosk appliance: fullscreen by default. Opt out only with --windowed
  // (the old opt-in --kiosk flag is still honoured). The launcher
  // (src/main.py) always passes --kiosk on the robot.
  const windowed = process.argv.includes('--windowed');
  const isKiosk = !windowed;

  // Fit whatever display is actually attached (7" LCD is 800x480, but X
  // may report a smaller mode). The old hardcoded 800x480 minimum left
  // the window unmapped (black screen) on smaller screens.
  let sw = 800;
  let sh = 480;
  try {
    const disp = screen.getPrimaryDisplay();
    if (disp && disp.workAreaSize) {
      sw = disp.workAreaSize.width || sw;
      sh = disp.workAreaSize.height || sh;
    }
  } catch (e) {
    console.error('[karma-ui] screen query failed, using 800x480:', e);
  }
  const winW = windowed ? Math.min(800, sw) : sw;
  const winH = windowed ? Math.min(480, sh) : sh;
  console.error(`[karma-ui] creating window ${winW}x${winH} kiosk=${isKiosk} screen=${sw}x${sh}`);

  mainWindow = new BrowserWindow({
    width: winW,
    height: winH,
    minWidth: Math.min(400, winW),
    minHeight: Math.min(300, winH),
    useContentSize: false,
    backgroundColor: '#000000',
    frame: false,
    fullscreen: isKiosk,
    kiosk: isKiosk,
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html')).catch((e) => {
    console.error('[karma-ui] loadFile failed:', e);
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
