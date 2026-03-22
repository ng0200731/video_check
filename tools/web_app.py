"""Flask web app for label inspection.

Routes:
  /            – main page
  /start       – POST: begin processing
  /progress    – GET: JSON progress
  /results     – GET: paginated results
  /img/<path>  – serve label images
"""

import os
import sqlite3
import threading

from flask import Flask, jsonify, render_template_string, request, send_file

from processor import (
    BASE_DIR,
    DB_PATH,
    APPROVED_DIR,
    REJECTED_DIR,
    run,
    state,
)

app = Flask(__name__)

VIDEO_PATH = os.path.join(BASE_DIR, "WhatsApp Video 2026-03-21 at 18.24.08.mp4")
REFERENCE_PATH = os.path.join(BASE_DIR, "IMG20260207170336.jp99.jpeg")

# ---------------------------------------------------------------------------
# Templates (inline, black & white per CLAUDE.md)
# ---------------------------------------------------------------------------

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label Inspection</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:monospace;background:#fff;color:#000;padding:24px;max-width:960px;margin:auto}
h1{font-size:1.4rem;margin-bottom:16px;border-bottom:2px solid #000;padding-bottom:8px}
button{font-family:monospace;font-size:.9rem;padding:8px 20px;border:2px solid #000;
  background:#fff;color:#000;cursor:pointer}
button:hover{background:#000;color:#fff}
button:disabled{opacity:.4;cursor:default;background:#fff;color:#000}
.bar-wrap{border:1px solid #000;height:24px;margin:12px 0;position:relative}
.bar-fill{height:100%;background:#000;transition:width .3s}
.bar-text{position:absolute;top:3px;left:8px;font-size:.75rem;color:#fff;mix-blend-mode:difference}
.stats{display:flex;gap:24px;margin:12px 0;font-size:.85rem}
.stats span{display:inline-block;padding:4px 8px;border:1px solid #000}
#status{margin:8px 0;font-size:.85rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin-top:16px}
.card{border:1px solid #ccc;padding:8px;text-align:center}
.card img{width:100%;height:auto;image-rendering:pixelated}
.card .meta{font-size:.7rem;margin-top:4px}
.card.approved{border-color:#000}
.card.rejected{border-color:#999}
.tag{display:inline-block;padding:1px 6px;font-size:.7rem;border:1px solid}
.tag.approved{border-color:#000;color:#000}
.tag.rejected{border-color:#999;color:#999}
.filter{margin:12px 0;font-size:.85rem}
.filter a{margin-right:12px;color:#000;text-decoration:none;border-bottom:1px solid transparent}
.filter a.active{border-bottom:1px solid #000;font-weight:bold}
.hidden{display:none}
</style>
</head>
<body>
<h1>Label Inspection</h1>

<div id="controls">
  <button id="btnStart" onclick="startProcessing()">Start Inspection</button>
</div>

<div id="progress" class="hidden">
  <div id="status">Initialising...</div>
  <div class="bar-wrap"><div class="bar-fill" id="bar" style="width:0%"></div>
    <div class="bar-text" id="barText">0%</div></div>
  <div class="stats">
    <span>Found: <b id="sFound">0</b></span>
    <span>Approved: <b id="sApproved">0</b></span>
    <span>Rejected: <b id="sRejected">0</b></span>
  </div>
</div>

<div id="results" class="hidden">
  <div class="filter">
    <a href="#" class="active" data-f="all" onclick="setFilter('all',this)">All</a>
    <a href="#" data-f="approved" onclick="setFilter('approved',this)">Approved</a>
    <a href="#" data-f="rejected" onclick="setFilter('rejected',this)">Rejected</a>
  </div>
  <div class="grid" id="grid"></div>
</div>

<script>
let poll=null, currentFilter='all';

function startProcessing(){
  document.getElementById('btnStart').disabled=true;
  document.getElementById('progress').classList.remove('hidden');
  fetch('/start',{method:'POST'}).then(()=>{
    poll=setInterval(checkProgress,800);
  });
}

function checkProgress(){
  fetch('/progress').then(r=>r.json()).then(d=>{
    let pct=0, total=d.total_frames||1;
    if(d.status==='scanning'){
      pct=Math.round(d.current_frame/total*50);
      document.getElementById('status').textContent='Scanning markers... frame '+d.current_frame+'/'+total;
    }else if(d.status==='capturing'){
      let found=d.labels_found||1;
      pct=50+Math.round(d.current_frame/found*50);
      document.getElementById('status').textContent='Capturing labels... '+d.current_frame+'/'+found;
    }else if(d.status==='done'){
      pct=100;
      document.getElementById('status').textContent='Done!';
      clearInterval(poll);
      loadResults();
    }else if(d.status==='error'){
      document.getElementById('status').textContent='Error: '+d.error;
      clearInterval(poll);
    }
    document.getElementById('bar').style.width=pct+'%';
    document.getElementById('barText').textContent=pct+'%';
    document.getElementById('sFound').textContent=d.labels_found;
    document.getElementById('sApproved').textContent=d.labels_approved;
    document.getElementById('sRejected').textContent=d.labels_rejected;
  });
}

function loadResults(){
  document.getElementById('results').classList.remove('hidden');
  fetch('/results?filter='+currentFilter).then(r=>r.json()).then(rows=>{
    let g=document.getElementById('grid');
    g.innerHTML='';
    rows.forEach(r=>{
      let d=document.createElement('div');
      d.className='card '+r.status;
      d.innerHTML='<img src="/img/'+encodeURIComponent(r.image_path)+'"><div class="meta">'
        +r.name+' <span class="tag '+r.status+'">'+r.status.toUpperCase()+'</span><br>'
        +'sim: '+r.similarity+'%</div>';
      g.appendChild(d);
    });
  });
}

function setFilter(f,el){
  currentFilter=f;
  document.querySelectorAll('.filter a').forEach(a=>a.classList.remove('active'));
  el.classList.add('active');
  loadResults();
  return false;
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/start", methods=["POST"])
def start():
    if state.status in ("scanning", "capturing"):
        return jsonify({"ok": False, "msg": "Already running"}), 409
    t = threading.Thread(target=run, args=(VIDEO_PATH, REFERENCE_PATH), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/progress")
def progress():
    return jsonify(state.to_dict())


@app.route("/results")
def results():
    filt = request.args.get("filter", "all")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if filt in ("approved", "rejected"):
        rows = conn.execute(
            "SELECT * FROM labels WHERE status=? ORDER BY name", (filt,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM labels ORDER BY name").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/img/<path:filepath>")
def serve_img(filepath):
    # filepath is the absolute path stored in the DB
    if os.path.isfile(filepath):
        return send_file(filepath, mimetype="image/png")
    return "Not found", 404


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
