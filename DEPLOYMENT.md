# WebModelDelivery — Production Server Deployment Guide

## Overview

The test harnesses require specific HTTP headers and server configuration to function correctly. Without these, you'll get silent failures: multi-threading won't activate, the Service Worker won't register, or `.wasm` files won't load.

### Required Features (in order of importance)

| Feature | Why | What breaks without it |
|---------|-----|----------------------|
| **HTTPS** | Service Workers require secure context | SW won't register at all (except localhost) |
| **COOP/COEP headers** | Enable `SharedArrayBuffer` for multi-thread WASM | Falls back to single-thread (~2-4× slower) |
| **`Service-Worker-Allowed: /`** | SW registered at root scope from subpath | SW scope limited, can't intercept model fetches |
| **MIME: `.wasm` → `application/wasm`** | Browser requires correct MIME for `WebAssembly.instantiate` | WASM fails to compile |
| **MIME: `.mjs` → `text/javascript`** | ES module imports require JS MIME | ONNX Runtime / Transformers.js won't load |
| **CORS headers** | CDN-hosted model files fetched cross-origin | Model downloads blocked |
| **Large request body / timeout** | Model files are 500MB–3GB | Downloads time out or get truncated |

### The Three Critical Headers

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless
Service-Worker-Allowed: /
```

**Why `credentialless` and not `require-corp`?** The harnesses load third-party scripts (Alpine.js, Tailwind CSS, CDN WASM files). `require-corp` would require every cross-origin resource to send a `Cross-Origin-Resource-Policy` header — most CDNs don't. `credentialless` is the modern alternative that enables `SharedArrayBuffer` while allowing cross-origin loads without CORP headers. It's supported in Chrome 96+, Firefox 119+, and Safari 15.2+.

---

## Nginx

### Minimal Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate     /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    root /var/www/webmodeldelivery;
    index index.html;

    # ── Critical: COOP/COEP for SharedArrayBuffer (multi-thread WASM) ──
    add_header Cross-Origin-Opener-Policy  "same-origin" always;
    add_header Cross-Origin-Embedder-Policy "credentialless" always;

    # ── Critical: Allow Service Worker to control root scope ──
    add_header Service-Worker-Allowed "/" always;

    # ── CORS (needed if model files are served from this origin) ──
    add_header Access-Control-Allow-Origin "*" always;
    add_header Access-Control-Allow-Methods "GET, OPTIONS" always;
    add_header Access-Control-Allow-Headers "Range, Content-Type" always;

    # ── MIME types (add if not in system mime.types) ──
    types {
        application/wasm                wasm;
        text/javascript                 mjs;
        application/octet-stream        gguf onnx model;
        application/json                json;
    }

    # ── model-sw.js must never be cached (SW update detection) ──
    location = /model-sw.js {
        add_header Cache-Control "no-cache, no-store, must-revalidate" always;
        # Re-add COOP/COEP — add_header in location blocks replaces parent
        add_header Cross-Origin-Opener-Policy  "same-origin" always;
        add_header Cross-Origin-Embedder-Policy "credentialless" always;
        add_header Service-Worker-Allowed "/" always;
    }

    # ── Large model files: increase timeouts and disable buffering ──
    location /models/ {
        client_max_body_size 0;           # No upload limit
        proxy_buffering off;              # Stream large responses
        proxy_read_timeout 3600s;         # 1 hour for slow connections
        # Re-add headers (location block overrides parent)
        add_header Cross-Origin-Opener-Policy  "same-origin" always;
        add_header Cross-Origin-Embedder-Policy "credentialless" always;
        add_header Access-Control-Allow-Origin "*" always;
    }

    # ── Static file caching (not SW, not HTML) ──
    location ~* \.(js|mjs|wasm|css|png|svg|ico)$ {
        expires 7d;
        add_header Cache-Control "public, immutable" always;
        # Re-add COOP/COEP
        add_header Cross-Origin-Opener-Policy  "same-origin" always;
        add_header Cross-Origin-Embedder-Policy "credentialless" always;
    }
}
```

> **Nginx gotcha:** `add_header` in a `location` block **replaces** all `add_header` directives from the parent block. You must re-add COOP/COEP in every `location` that has its own `add_header`. Alternatively, use the [`headers-more` module](https://github.com/openresty/headers-more-nginx-module) with `more_set_headers` which doesn't have this limitation.

### Verify

```bash
curl -sI https://example.com/ | grep -iE 'cross-origin|service-worker'
# Should show:
#   cross-origin-opener-policy: same-origin
#   cross-origin-embedder-policy: credentialless
#   service-worker-allowed: /

curl -sI https://example.com/model-sw.js | grep -i content-type
# Should show: text/javascript (not application/octet-stream)

curl -sI https://example.com/path/to/file.wasm | grep -i content-type
# Should show: application/wasm
```

---

## Apache (.htaccess or httpd.conf)

### .htaccess (shared hosting friendly)

```apache
# ── Enable required modules (if not already enabled) ──
# On shared hosting these are usually already active.
# On a dedicated server, run:  a2enmod headers rewrite mime

# ── Critical: COOP/COEP for SharedArrayBuffer ──
<IfModule mod_headers.c>
    Header always set Cross-Origin-Opener-Policy "same-origin"
    Header always set Cross-Origin-Embedder-Policy "credentialless"
    Header always set Service-Worker-Allowed "/"
    Header always set Access-Control-Allow-Origin "*"
    Header always set Access-Control-Allow-Methods "GET, OPTIONS"
    Header always set Access-Control-Allow-Headers "Range, Content-Type"
</IfModule>

# ── MIME types ──
<IfModule mod_mime.c>
    AddType application/wasm .wasm
    AddType text/javascript .mjs
    AddType application/octet-stream .gguf .onnx .model
</IfModule>

# ── Prevent caching of Service Worker file ──
<Files "model-sw.js">
    <IfModule mod_headers.c>
        Header set Cache-Control "no-cache, no-store, must-revalidate"
    </IfModule>
</Files>

# ── Handle OPTIONS preflight for CORS ──
<IfModule mod_rewrite.c>
    RewriteEngine On
    RewriteCond %{REQUEST_METHOD} OPTIONS
    RewriteRule ^(.*)$ $1 [R=204,L]
</IfModule>
```

### Verify

Same `curl` commands as Nginx above. If `.htaccess` isn't working, check that `AllowOverride All` (or at least `AllowOverride FileInfo Options`) is set in the Apache vhost config:

```apache
<Directory /var/www/webmodeldelivery>
    AllowOverride All
</Directory>
```

---

## IIS (Windows / .NET Framework)

### web.config

Place this `web.config` in the site root:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>

  <system.webServer>

    <!-- ── Critical: COOP/COEP + SW headers ── -->
    <httpProtocol>
      <customHeaders>
        <add name="Cross-Origin-Opener-Policy" value="same-origin" />
        <add name="Cross-Origin-Embedder-Policy" value="credentialless" />
        <add name="Service-Worker-Allowed" value="/" />
        <add name="Access-Control-Allow-Origin" value="*" />
        <add name="Access-Control-Allow-Methods" value="GET, OPTIONS" />
        <add name="Access-Control-Allow-Headers" value="Range, Content-Type" />
      </customHeaders>
    </httpProtocol>

    <!-- ── MIME types ── -->
    <staticContent>
      <!-- Remove first to avoid "duplicate" errors if already defined -->
      <remove fileExtension=".wasm" />
      <mimeMap fileExtension=".wasm" mimeType="application/wasm" />

      <remove fileExtension=".mjs" />
      <mimeMap fileExtension=".mjs" mimeType="text/javascript" />

      <remove fileExtension=".gguf" />
      <mimeMap fileExtension=".gguf" mimeType="application/octet-stream" />

      <remove fileExtension=".onnx" />
      <mimeMap fileExtension=".onnx" mimeType="application/octet-stream" />

      <remove fileExtension=".json" />
      <mimeMap fileExtension=".json" mimeType="application/json" />
    </staticContent>

    <!-- ── Large file support ── -->
    <requestFiltering>
      <!-- 4GB max (in bytes).  Default is 30MB which will block model files -->
      <requestLimits maxAllowedContentLength="4294967296" />
    </requestFiltering>

    <!-- ── Prevent caching of model-sw.js ── -->
    <location path="model-sw.js">
      <system.webServer>
        <staticContent>
          <clientCache cacheControlMode="DisableCache" />
        </staticContent>
      </system.webServer>
    </location>

  </system.webServer>

  <!-- ── ASP.NET max request size (if using .NET pipeline) ── -->
  <system.web>
    <httpRuntime maxRequestLength="4194304" />  <!-- 4GB in KB -->
  </system.web>

</configuration>
```

### IIS Manager GUI (alternative)

If you prefer the GUI over web.config:

1. **MIME Types**: IIS Manager → site → MIME Types → Add `.wasm` = `application/wasm`, `.mjs` = `text/javascript`
2. **HTTP Response Headers**: IIS Manager → site → HTTP Response Headers → Add the three COOP/COEP/SW headers
3. **Request Filtering**: IIS Manager → site → Request Filtering → Edit Feature Settings → set Max Content Length to 4294967296

### IIS Common Issues

- **`.mjs` returns 404**: IIS doesn't serve unknown extensions by default. The MIME mapping above fixes this.
- **`.wasm` returns 404 or wrong type**: Same — must be registered as `application/wasm`.
- **Large files fail at 30MB**: Default `maxAllowedContentLength` is 30MB. The web.config above raises it to 4GB.
- **Handler conflicts**: If you have ASP.NET or URL Rewrite rules, ensure they don't intercept `/model-sw.js` or `/models/*` paths.

---

## GoDaddy / Plesk Hosting

### GoDaddy Linux (cPanel / Plesk)

**Good news:** Apache `.htaccess` works on most GoDaddy Linux plans.

1. Upload all files to `public_html/` (or a subdirectory)
2. Create `.htaccess` in the same directory using the Apache config above
3. **Verify HTTPS**: GoDaddy provides free SSL on most plans. Ensure your site loads via `https://`. Service Workers require HTTPS.
4. **File size limits**: GoDaddy shared hosting may have PHP `upload_max_filesize` / `post_max_size` limits, but these only apply to uploads, not static file serving. Model files served as static downloads should work.

**Plesk-specific (Linux):**

- Plesk → Domains → your domain → Apache & nginx Settings
- Under "Additional directives for HTTP" and "Additional directives for HTTPS", add:

```apache
<IfModule mod_headers.c>
    Header always set Cross-Origin-Opener-Policy "same-origin"
    Header always set Cross-Origin-Embedder-Policy "credentialless"
    Header always set Service-Worker-Allowed "/"
</IfModule>
```

- Alternatively, Plesk allows `.htaccess` — just upload it as described above.

**Potential GoDaddy Linux issues:**

- `mod_headers` not enabled → Headers won't apply. Contact GoDaddy support or check cPanel → "Apache Handlers". Most plans have it.
- Shared hosting connection timeouts → Large models (>1GB) may time out on slow connections. The SW handles this gracefully with retry, but initial cold loads could fail. Consider a CDN for model files.

### GoDaddy Windows (Plesk / Classic ASP.NET)

**Use the `web.config` approach** from the IIS section above.

1. Upload all files to the site root via Plesk File Manager or FTP
2. Upload `web.config` to the site root
3. **Verify HTTPS** is active on your GoDaddy Windows plan

**Plesk-specific (Windows):**

- Plesk → Domains → your domain → Hosting Settings → Ensure "SSL/TLS support" is checked
- Plesk → Domains → your domain → IIS Settings:
  - Under "Custom HTTP Headers", add:
    - `Cross-Origin-Opener-Policy: same-origin`
    - `Cross-Origin-Embedder-Policy: credentialless`
    - `Service-Worker-Allowed: /`
  - Under "MIME Types", add `.wasm` → `application/wasm` and `.mjs` → `text/javascript`

**Potential GoDaddy Windows issues:**

- **`web.config` overwritten**: Plesk may regenerate `web.config`. Use Plesk IIS Settings GUI for headers instead.
- **Handler mappings**: Windows hosting with ASP.NET may intercept extensionless URLs or `.mjs`. If `.mjs` files 404, try adding a handler in web.config:

```xml
<system.webServer>
  <handlers>
    <add name="MjsHandler" path="*.mjs" verb="GET" type="System.Web.StaticFileHandler" />
  </handlers>
</system.webServer>
```

- **Request filtering in shared hosting**: GoDaddy may enforce their own `maxAllowedContentLength`. If model files fail to load, you may need VPS/dedicated hosting for files >100MB.

---

## CDN Configuration (Recommended for Model Files)

For production, serve model files from a CDN rather than your web server. The Service Worker fetches shards from the CDN origin and reassembles them as same-origin responses (COEP-safe).

### How it works

The harness architecture is already designed for this:

```
Browser → fetch(/models/prefix/model.gguf)
       → Service Worker intercepts
       → SW fetches shards from CDN (cdnBase in MODEL_SOURCES)
       → SW reassembles as same-origin Response
       → Browser receives COEP-safe response
```

You only need the COOP/COEP headers on **your web server** (where the HTML/JS lives). The CDN just needs standard CORS:

```
Access-Control-Allow-Origin: *
```

### CDN providers that work out of the box

- **Cloudflare R2** — free egress, CORS by default
- **AWS CloudFront + S3** — configure CORS on the S3 bucket
- **Bunny CDN** — CORS enabled in pull zone settings
- **jsDelivr / unpkg** — already configured (used for wllama WASM)

### Recommended setup

| Content | Served from | Headers needed |
|---------|------------|----------------|
| HTML, JS, CSS, model-sw.js | Your web server | COOP, COEP, SW-Allowed |
| `.wasm` files (wllama, ONNX) | CDN or your server | CORS (`Access-Control-Allow-Origin: *`) |
| Model files (`.gguf` shards, `.onnx`) | CDN | CORS only (SW proxies to same-origin) |
| `filemap.json` | Same origin as model files | CORS |

---

## Verification Checklist

After deploying, open browser DevTools Console and check:

```javascript
// 1. SharedArrayBuffer available? (requires COOP+COEP)
console.log('SharedArrayBuffer:', typeof SharedArrayBuffer !== 'undefined');
// Should be: true

// 2. Cross-origin isolated? (confirms COOP+COEP working)
console.log('crossOriginIsolated:', crossOriginIsolated);
// Should be: true

// 3. Service Worker registered?
const reg = await navigator.serviceWorker.getRegistration('/');
console.log('SW:', reg?.active?.state);
// Should be: "activated"
```

If `crossOriginIsolated` is `false`, wllama falls back to single-thread (works but ~2-4× slower).

### Quick diagnostic for common failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| SW won't register | Not HTTPS | Enable SSL |
| SW won't register | `model-sw.js` returns wrong MIME | Ensure `.js` → `text/javascript` |
| `SharedArrayBuffer is not defined` | Missing COOP/COEP headers | Add both headers |
| `crossOriginIsolated` is `false` | COEP blocks a resource | Check for `require-corp` → use `credentialless` |
| WASM won't compile | `.wasm` MIME wrong | Map `.wasm` → `application/wasm` |
| Transformers.js won't load | `.mjs` returns 404 or wrong MIME | Map `.mjs` → `text/javascript` |
| Model download stalls/fails | File size limit or timeout | Raise limits (see server configs above) |
| Model download CORS error | CDN missing CORS headers | Add `Access-Control-Allow-Origin: *` to CDN |

---

## File Layout

Minimum files to deploy:

```
/                           ← site root
├── model-sw.js             ← Service Worker (MUST be at root for scope)
├── harness-llm-index.html  ← (or whichever harness)
├── harness-index.html
├── harness-unified.html
├── .htaccess               ← Apache (or web.config for IIS)
└── (model files served from CDN, not necessarily here)
```

**`model-sw.js` must be at the root** of the domain (or at least at the same path level as the `scope` it registers with). A Service Worker can only control pages at or below its own URL path.
