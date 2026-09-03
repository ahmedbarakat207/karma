import http.server, socketserver, os, sys

PORT = 8787
BODY_DIR = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=BODY_DIR, **kw)
    def log_message(self, fmt, *args):
        pass

if __name__ == "__main__":
    print(f"Assembly viewer running at http://localhost:{PORT}/assembly_viewer.html")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
