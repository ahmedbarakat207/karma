const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

let mainWindow = null;

function createWindow() {
  const isKiosk = process.argv.includes('--kiosk') || process.env.KARMA_KIOSK === '1';

  mainWindow = new BrowserWindow({
    width: 800,
    height: 480,
    minWidth: 800,
    minHeight: 480,
    useContentSize: true,
    backgroundColor: '#000000',
    frame: false,
    fullscreen: isKiosk,
    kiosk: isKiosk,
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false
    }
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html'));

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
