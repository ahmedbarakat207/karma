"""
assembly_server.py
==================
Tiny HTTP server that serves the body/ STL files and the assembly viewer HTML.
Run with:  python assembly_server.py
Then open  http://localhost:8787/assembly_viewer.html
"""
import http.server, socketserver, os, sys

PORT = 8787
BODY_DIR = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=BODY_DIR, **kw)
    def log_message(self, fmt, *args):
        pass   # suppress request logs

print(f"Assembly viewer running at  http://localhost:{PORT}/assembly_viewer.html")
print("Press Ctrl-C to stop.")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
