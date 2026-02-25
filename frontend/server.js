const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3000;
const PUBLIC_DIR = 'public'; // Your HTML files are in /public

const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.txt': 'text/plain'
};

const server = http.createServer((req, res) => {
  console.log(`${req.method} ${req.url}`);
  
  // Remove query parameters
  let pathname = req.url.split('?')[0];
  
  // Default to index.html
  if (pathname === '/') {
    pathname = '/index.html';
  }
  
  // Get the full file path (relative to public directory)
  const filePath = path.join(__dirname, PUBLIC_DIR, pathname);
  const ext = path.extname(filePath);
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';
  
  console.log(`Looking for file: ${filePath}`);
  
  // Check if file exists
  fs.access(filePath, fs.constants.F_OK, (err) => {
    if (err) {
      console.log(`File not found: ${filePath}`);
      
      // If requesting root and index.html doesn't exist
      if (pathname === '/index.html') {
        const fullPublicPath = path.join(__dirname, PUBLIC_DIR);
        
        // Check if public directory exists
        fs.access(fullPublicPath, fs.constants.F_OK, (dirErr) => {
          if (dirErr) {
            console.log(`Public directory not found: ${fullPublicPath}`);
            
            const errorHTML = `
            <!DOCTYPE html>
            <html>
            <head>
                <title>ADS-B Dashboard - Configuration Error</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background: linear-gradient(135deg, #0a0f1e 0%, #1a1f2e 100%);
                        color: white;
                        margin: 0;
                        padding: 40px;
                        text-align: center;
                    }
                    .container {
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 40px;
                        background: rgba(26, 31, 46, 0.8);
                        border-radius: 15px;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                    }
                    h1 {
                        color: #ff4444;
                        margin-bottom: 20px;
                    }
                    .error {
                        background: rgba(220, 53, 69, 0.2);
                        border-left: 4px solid #dc3545;
                        padding: 15px;
                        margin: 20px 0;
                        border-radius: 8px;
                        text-align: left;
                    }
                    code {
                        background: rgba(0, 0, 0, 0.3);
                        padding: 2px 6px;
                        border-radius: 4px;
                        font-family: monospace;
                        color: #ffc107;
                    }
                    .solution {
                        background: rgba(25, 135, 84, 0.2);
                        border-left: 4px solid #198754;
                        padding: 15px;
                        margin: 20px 0;
                        border-radius: 8px;
                        text-align: left;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🚨 Configuration Error</h1>
                    <p>Server is running, but <code>public</code> directory was not found.</p>
                    
                    <div class="error">
                        <strong>❌ Problem:</strong> 
                        <p>The server is looking for files in <code>${fullPublicPath}</code></p>
                        <p>But this directory does not exist.</p>
                    </div>
                    
                    <div class="solution">
                        <strong>✅ Solution:</strong>
                        <p>Your file structure should be:</p>
                        <pre style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; overflow: auto;">
frontend/
├── server.js
├── package.json
└── public/           ← This directory is missing
    └── index.html    ← Your HTML file should be here</pre>
                        
                        <p><strong>Quick fix:</strong> Move your <code>index.html</code> into a <code>public</code> folder:</p>
                        <pre style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px;">
cd frontend
mkdir public
move index.html public/  # On Windows
# or on Mac/Linux: mv index.html public/</pre>
                    </div>
                    
                    <div style="margin-top: 30px; padding: 15px; background: rgba(13, 110, 253, 0.1); border-radius: 8px;">
                        <h3>📁 Current Directory Structure:</h3>
                        <pre style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; overflow: auto; text-align: left;">
${getDirectoryTree(__dirname)}</pre>
                    </div>
                </div>
            </body>
            </html>`;
            
            res.writeHead(200, { 
              'Content-Type': 'text/html',
              'Access-Control-Allow-Origin': '*'
            });
            res.end(errorHTML, 'utf-8');
          } else {
            // Public directory exists but index.html doesn't
            res.writeHead(404, { 
              'Content-Type': 'text/html',
              'Access-Control-Allow-Origin': '*'
            });
            res.end(`
            <html>
            <head><title>404 - File Not Found</title></head>
            <body style="font-family: Arial; padding: 40px; background: #0a0f1e; color: white;">
              <h1>404 - index.html not found</h1>
              <p>The file <code>${filePath}</code> was not found.</p>
              <p>Make sure your HTML file is named <code>index.html</code> and placed in the <code>public</code> directory.</p>
            </body>
            </html>`, 'utf-8');
          }
        });
      } else {
        // For other files, send 404
        res.writeHead(404, { 'Content-Type': 'text/html' });
        res.end('<h1>404 Not Found</h1><p>The requested file was not found on this server.</p>', 'utf-8');
      }
      return;
    }
    
    // File exists, read and serve it
    fs.readFile(filePath, (err, data) => {
      if (err) {
        console.error(`Error reading file ${filePath}:`, err);
        res.writeHead(500);
        res.end(`Server Error: ${err.code}`);
        return;
      }
      
      console.log(`✅ Serving file: ${filePath}`);
      res.writeHead(200, { 
        'Content-Type': contentType,
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'no-cache, no-store, must-revalidate'
      });
      res.end(data, 'utf-8');
    });
  });
});

// Helper function to get directory tree
function getDirectoryTree(dir, prefix = '') {
  let result = '';
  const files = fs.readdirSync(dir);
  
  files.forEach((file, index) => {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    const isLast = index === files.length - 1;
    
    result += prefix + (isLast ? '└── ' : '├── ') + file + '\n';
    
    if (stat.isDirectory()) {
      result += getDirectoryTree(fullPath, prefix + (isLast ? '    ' : '│   '));
    }
  });
  
  return result;
}

server.listen(PORT, () => {
  console.log('='.repeat(50));
  console.log('🚀 ADS-B Anomaly Detection Frontend Server');
  console.log('='.repeat(50));
  console.log(`📡 Server running at http://localhost:${PORT}`);
  console.log(`📁 Serving from: ${path.join(__dirname, PUBLIC_DIR)}`);
  console.log(`💡 Make sure backend is running at http://localhost:8000`);
  console.log('='.repeat(50));
  console.log('\n📋 Available URLs:');
  console.log(`   Dashboard: http://localhost:${PORT}`);
  console.log(`   Backend API: http://localhost:8000`);
  console.log(`   API Docs: http://localhost:8000/api/docs`);
  console.log(`   WebSocket: ws://localhost:8000/ws`);
  console.log('='.repeat(50));
  
  // Check if public directory exists
  const publicPath = path.join(__dirname, PUBLIC_DIR);
  fs.access(publicPath, fs.constants.F_OK, (err) => {
    if (err) {
      console.log(`\n⚠️  WARNING: ${PUBLIC_DIR}/ directory not found at: ${publicPath}`);
      console.log(`   The server will look for files in this directory.`);
      console.log(`   Create it and put your index.html inside.`);
    } else {
      console.log(`\n✅ ${PUBLIC_DIR}/ directory found at: ${publicPath}`);
      console.log('\n📁 Files in public directory:');
      fs.readdir(publicPath, (err, files) => {
        if (err) {
          console.error('Error reading public directory:', err);
        } else {
          files.forEach(file => {
            const filePath = path.join(publicPath, file);
            try {
              const stat = fs.statSync(filePath);
              const type = stat.isDirectory() ? '📁' : '📄';
              console.log(`   ${type} ${file}`);
            } catch (e) {
              console.log(`   ❓ ${file} (error reading)`);
            }
          });
        }
        console.log('='.repeat(50));
      });
    }
  });
});