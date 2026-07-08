#!/usr/bin/env python3
# ipcam_yolo4_final.py
# Single-file app: live YOLO detections -> detections_log.csv
# Analytics: Google-style "Popular Times" (09:00-21:00), current hour highlighted (vertical bars)
# Alerts persisted to CSV and shown on Home

import os, time, gc, uuid, math, hashlib
from collections import deque, defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, Response, request, redirect, url_for, jsonify, send_file, make_response, render_template_string
from werkzeug.utils import secure_filename

# added from top------main script from Here working fine
from app.core.logger import (
    ensure_log_headers,
    write_log_row
)

from app.ai_engine.recognition.face import (
    face_manager
)

from app.services.alert_service import (
    alert_service
)
from app.camera.stream import (
    open_capture,
    blank_jpeg
)
from app.ai_engine.analytics.behavior import (
    point_in_poly,
    bbox_center,
    crosses_line,
    zone_of_point
)

try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import face_recognition
except Exception:
    face_recognition = None
# ---------------- CONFIG ----------------

from app.core.config import *

from app.core.logger import (
    ensure_log_headers,
    write_log_row
)

from app.ai_engine.recognition.face import (
    face_manager
)
os.makedirs(KNOWN_IMAGES_DIR, exist_ok=True)

# ---------- faces csv ----------

face_manager.ensure_faces_csv()
# ---------- load models ----------
from app.ai_engine.model_manager import (
    model_manager
)
# below line for test
alert_service.add_alert(
    "test",
    "Test alert generated during startup"
)
#above lines for test
# ---------- known faces ----------

known_encodings, known_meta = (
    face_manager.load_known_faces()
)
# ---------- logging ----------

ensure_log_headers()

model_manager.load_models()

model = model_manager.yolo

age_net = model_manager.age_net

gender_net = model_manager.gender_net

# ---------- helpers ----------
#added this code to behaviour.py
# ---------- state ----------
last_detections = []
entered_hourly = {}
exited_hourly = {}
tracks = {}
next_tid = 1

# ---------- capture & detection ----------

def generate_frames():
    global last_detections, tracks, next_tid, entered_hourly, exited_hourly, alerts
    cap = open_capture(IP_CAMERA_URL)
    if cap is None or not cap.isOpened():
        while True:
            yield blank_jpeg("Camera not available")
            time.sleep(1.0)
    miss = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            miss += 1
            if miss >= 10:
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(0.5)
                cap = open_capture(IP_CAMERA_URL)
                miss = 0
                if cap is None or not cap.isOpened():
                    yield blank_jpeg("Reconnecting...")
                    time.sleep(1.0)
                    continue
            else:
                yield blank_jpeg("Reconnecting...")
                time.sleep(0.2)
                continue
        else:
            miss = 0

        detections = []
        det_list = []
        results = None
        if model is not None:
            try:
                results = model(frame, verbose=False)
            except Exception as e:
                print("YOLO inference error:", e)
                results = None
        boxes = None
        if results and len(results)>0:
            try:
                boxes = results[0].boxes
            except Exception:
                boxes = None
        if boxes:
            for box in boxes:
                try:
                    cls = int(box.cls[0])
                    name_cls = model.names[cls] if (hasattr(model,'names') and cls in model.names) else str(cls)
                    if str(name_cls).lower() != "person":
                        continue
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    if x2<=x1 or y2<=y1:
                        continue
                    crop_bgr = frame[y1:y2, x1:x2]
                    if crop_bgr.size == 0:
                        continue
                    person_name = "Unknown"
                    cats = []
                    face_roi = None
                    if face_recognition is not None:
                        try:
                            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                            locs = face_recognition.face_locations(crop_rgb, number_of_times_to_upsample=FACE_UPSAMPLE, model=FACE_DETECTOR_MODEL)
                            if locs:
                                encs = face_recognition.face_encodings(crop_rgb, known_face_locations=locs, num_jitters=ENCODER_JITTERS, model=ENCODER_MODEL)
                                if encs and known_encodings:
                                    dists = face_recognition.face_distance(known_encodings, encs[0])
                                    if len(dists)>0 and float(np.min(dists)) < FACE_MATCH_THRESHOLD:
                                        idx = int(np.argmin(dists))
                                        person_name = known_meta[idx]['name']
                                        cats = known_meta[idx]['categories']
                                top,right,bottom,left = locs[0]
                                h,w = crop_bgr.shape[:2]
                                pad_y = int(PAD_RATIO * (bottom-top)); pad_x = int(PAD_RATIO * (right-left))
                                t = max(0, top-pad_y); b2 = min(h, bottom+pad_y); l = max(0, left-pad_x); r2 = min(w, right+pad_x)
                                if b2>t and r2>l:
                                    face_roi = crop_bgr[t:b2, l:r2].copy()
                        except Exception:
                            pass
                    age_bucket, gender, gender_conf = None, None, None
                    if face_roi is not None and face_roi.size>0 and min(face_roi.shape[:2]) >= MIN_FACE_SIDE_PX:
                        try:
                            if age_net is not None:
                                blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227,227), (78,87,114), swapRB=False)
                                age_net.setInput(blob)
                                a_scores = age_net.forward()[0]
                                age_bucket = AGE_LIST[int(np.argmax(a_scores))]
                        except Exception:
                            age_bucket = None
                        try:
                            if gender_net is not None:
                                blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227,227), (78,87,114), swapRB=False)
                                gender_net.setInput(blob)
                                g_scores = gender_net.forward()[0]
                                g_idx = int(np.argmax(g_scores))
                                gender_conf = float(g_scores[g_idx])
                                gender_raw = GENDER_LIST[g_idx]
                                gender = gender_raw if gender_conf >= GENDER_CONF_THRESH else "Unknown"
                        except Exception:
                            gender, gender_conf = None, None

                    parts = [person_name] if person_name else ["Unknown"]
                    if gender: parts.append(gender)
                    if age_bucket: parts.append(age_bucket)
                    label = " ".join(parts)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, label, (x1, max(y1-6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
                    z = zone_of_point(bbox_center((x1,y1,x2,y2)))
                    det = {"name": person_name, "categories": cats, "zone": z, "img": crop_bgr.copy(), "face": face_roi, "age": age_bucket, "gender": gender, "bbox": (x1,y1,x2,y2)}
                    detections.append(det)
                    write_log_row({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "name": person_name,
                        "categories": ";".join(cats) if cats else "",
                        "zone": z,
                        "alert": "",
                        "gender": gender or "",
                        "gender_conf": f"{gender_conf:.2f}" if gender_conf else "",
                        "age_bucket": age_bucket or "",
                        "detection_type": "person"
                    })
                    cats_norm = [str(c).strip().lower() for c in cats] if cats else []
                    if "vip" in cats_norm:
                        info = f"VIP detected: {person_name} in {z}. Suggest priority assistance."
                        alert_service.add_alert("vip", info, zone=z, name=person_name)
                    if "highbuyer" in cats_norm:
                        info = f"High-buyer candidate: {person_name} in {z}."
                        alert_service.add_alert("highbuyer", info, zone=z, name=person_name)
                    cx,cy = bbox_center((x1,y1,x2,y2))
                    det_list.append({"center": (cx,cy), "bbox": (x1,y1,x2,y2), "gender": gender, "age": age_bucket})
                except Exception:
                    continue

        # tracking update
        t_now = time.time()
        for tid in list(tracks.keys()):
            if t_now - tracks[tid]["last_seen"] > 3.0:
                del tracks[tid]
        used = set()
        global next_tid
        for d in det_list:
            cx,cy = d["center"]
            best_tid, best_dist = None, 9999
            for tid, tr in tracks.items():
                px,py = tr["now"]
                dist = math.hypot(cx-px, cy-py)
                if dist < 120 and dist < best_dist and tid not in used:
                    best_dist, best_tid = dist, tid
            if best_tid is None:
                tid = next_tid; next_tid += 1
                tracks[tid] = {"last": (cx,cy), "now":(cx,cy), "last_seen": t_now, "gender": d.get("gender"), "age": d.get("age")}
                used.add(tid)
            else:
                tr = tracks[best_tid]
                tr["last"] = tr["now"]; tr["now"] = (cx,cy); tr["last_seen"] = t_now
                if not tr.get("gender") and d.get("gender"):
                    tr["gender"] = d.get("gender")
                if not tr.get("age") and d.get("age"):
                    tr["age"] = d.get("age")
                used.add(best_tid)

        # line crossing counts
        for tid, tr in tracks.items():
            if "last" in tr and "now" in tr:
                if crosses_line(tr["last"], tr["now"], COUNT_LINES["entry"]):
                    hk = datetime.now().strftime("%Y-%m-%d %H:00")
                    entered_hourly.setdefault(hk, {"total":0,"Male":0,"Female":0,"Unknown":0})
                    entered_hourly[hk]["total"] += 1
                if crosses_line(tr["last"], tr["now"], COUNT_LINES["exit"]):
                    hk = datetime.now().strftime("%Y-%m-%d %H:00")
                    exited_hourly.setdefault(hk, {"total":0,"Male":0,"Female":0,"Unknown":0})
                    exited_hourly[hk]["total"] += 1

        # queue & crowd detection
        queue_count = sum(1 for tr in tracks.values() if tr.get("now") and point_in_poly(tr["now"], ZONES["billing"]))
        crowd_count = sum(1 for tr in tracks.values() if tr.get("now") and point_in_poly(tr["now"], ZONES["crowd1"]))
        if queue_count > QUEUE_THRESH:
            alert_service.add_alert("queue_overflow", f"Queue overflow: {queue_count} persons at billing. Open extra counter.", zone="billing")
        if crowd_count > CROWD_THRESH:
            alert_service.add_alert("crowd", f"Crowd: {crowd_count} persons in crowd area. Check safety.", zone="crowd1")

        last_detections = detections[:16]
        if cv2 is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            yield blank_jpeg("No OpenCV")
        gc.collect()

# ---------- HTML (single-file SPA) ----------
HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>EDGE COMPUTE - Store Solution</title>
<style>
:root { --bg:#0b0f14; --panel:#141a22; --muted:#9aa4b2; --text:#e6edf3; --accent:#0ea5e9; --warn:#f59e0b; --alert:#ef4444; }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,Arial,sans-serif}
.nav{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;background:#0d131a;border-bottom:1px solid #1f2937}
.tabs a{color:#e5e7eb;padding:6px 10px;border-radius:8px;text-decoration:none;margin-right:6px}
.tabs a.active{background:#111827;border:1px solid #334155}
.container{display:grid;grid-template-columns:2fr 1fr;gap:14px;padding:14px}
.panel{background:#141a22;border:1px solid #1f2937;border-radius:12px;padding:12px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.alerts{display:flex;flex-direction:column;gap:8px;max-height:540px;overflow:auto}
.alert-item{background:#1a222d;border:1px solid #273244;border-left:4px solid var(--accent);padding:8px;border-radius:8px;color:#d1d5db}
.alert-item.red{border-left-color:var(--alert)}
.alert-item.warn{border-left-color:var(--warn)}
.row-cards{display:flex;gap:10px;flex-wrap:wrap}
.card{background:#121821;border:1px solid #1f2937;border-radius:10px;padding:8px;width:230px}
.small{font-size:12px;color:#9aa4b2}
.btn{background:var(--accent);color:#fff;border:none;padding:6px 10px;border-radius:8px;cursor:pointer;text-decoration:none;display:inline-block}
.btn.secondary{background:#1f2937;color:#e5e7eb;border:1px solid #334155}
.input{background:#0f1620;color:#e5e7eb;border:1px solid #334155;border-radius:8px;padding:6px 8px}
.face{width:100%;border-radius:6px;border:1px solid #1f2937;display:block}
.hidden{display:none !important}
.canvas-wrap{background:#0b1220;border:1px solid #1f2937;border-radius:10px;padding:8px}
/* === NEW: force consistent 16:9 ratio for camera cards === */
.zone-box { aspect-ratio: 16 / 9; background: #0b1220; border-radius: 10px; overflow: hidden; display: flex; flex-direction: column; justify-content: flex-start; }
.zone-box img { width: 100%; height: 100%; object-fit: cover; display: block; border-radius: 6px; }
/* LIVE tag styling for Popular Times header area (small badge) */
.live-badge { display:inline-block; padding:4px 6px; border-radius:6px; font-weight:600; background:#ff4d6d; color:#fff; font-size:12px; margin-right:8px; }
.subtitle { color:var(--muted); font-size:13px; margin-left:6px; }
</style>
</head>
<body>
<div class="nav">
  <div style="display:flex;align-items:center;gap:8px">
    <div style="width:26px;height:26px;border-radius:6px;background:linear-gradient(135deg,#0ea5e9,#6366f1)"></div>
    <div style="font-weight:600">EDGE COMPUTE - Store Solution</div>
  </div>
  <div class="tabs">
    <a href="#" class="active" onclick="showTab('home');return false;">Home</a>
    <a href="#" onclick="showTab('analytics');return false;">Analytics</a>
    <a href="#" onclick="showTab('inventory');return false;">Inventory</a>
    <a href="#" onclick="showTab('offers');return false;">Offers</a>
    <a href="#" onclick="showTab('dashboard');return false;">Dashboard</a>
    <a href="#" onclick="showTab('wifi');return false;">WiFi</a>
  </div>
</div>

<!-- Home -->
<div class="container" id="homeTab">
  <div>
    <div class="panel">
      <h2>Live View</h2>
      <div class="row3">
        <div class="zone-box"><div class="small">Cloth Section</div><img src="/video1" alt="live feed"/></div>
        <div class="zone-box"><div class="small">Entrance</div><div style="height:100%;display:flex;align-items:center;justify-content:center;border:1px dashed #334155;border-radius:10px;color:#94a3b8">Secondary</div></div>
        <div class="zone-box"><div class="small">Exit</div><div style="height:100%;display:flex;align-items:center;justify-content:center;border:1px dashed #334155;border-radius:10px;color:#94a3b8">Secondary</div></div>
      </div>
    </div>

    <div class="panel">
      <h2>Current Detections</h2>
      <div id="current_cards" class="row-cards"></div>
    </div>
  </div>

  <div>
    <div class="panel"><h2>Alerts</h2><div id="alerts" class="alerts"></div></div>

    <div class="panel">
      <h2>Manage Persons</h2>
      <form action="/upload_csv" method="post" enctype="multipart/form-data" style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <input type="file" name="file" class="input" accept=".csv" required>
        <button type="submit" class="btn">Upload faces.csv</button>
        <a class="btn secondary" href="/download_faces_csv">Download faces.csv</a>
      </form>
      <hr style="border:0;border-top:1px solid #1f2937;margin:10px 0"/>
      <form action="/upload_image" method="post" enctype="multipart/form-data" style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <input type="file" name="image" accept="image/*" class="input" required>
        <input type="text" name="name" placeholder="Name" class="input" required>
        <input type="text" name="categories" placeholder="vip;highbuyer" class="input">
        <button type="submit" class="btn">Upload image</button>
      </form>
    </div>
  </div>
</div>

<!-- Analytics -->
<div class="container hidden" id="analyticsTab" style="grid-template-columns:1fr 1fr;">
  <div class="panel" style="grid-column:1/-1">
    <h2 style="display:flex;align-items:center;gap:8px"><span class="live-badge">LIVE</span>Popular times <span class="subtitle" id="pt-sub">Not enough data yet</span></h2>
    <div style="display:flex;gap:8px;align-items:center">
      <label class="small">Select day:</label>
      <select id="day_select" onchange="renderPopularTimes()"></select>
      <button class="btn" onclick="renderPopularTimes()">Refresh</button>
      <span class="small" style="margin-left:8px">Popular Times (09:00 - 21:00)</span>
    </div>
  </div>

  <div class="panel" style="grid-column:1/-1">
    <div id="popular_times_div" class="canvas-wrap" style="height:420px"></div>
  </div>

  <div class="panel"><h3>Visitors by Zone</h3><div id="chart_zone" class="canvas-wrap" style="height:260px"></div></div>
  <div class="panel"><h3>Visitors by Gender / Age</h3><div id="chart_gender" class="canvas-wrap" style="height:260px"></div></div>
</div>

<!-- Placeholder -->
<div class="container hidden" id="inventoryTab"><div class="panel"><h2>Inventory Management</h2><p class="small">Placeholder</p></div></div>
<div class="container hidden" id="offersTab"><div class="panel"><h2>Offers Management</h2><p class="small">Placeholder</p></div></div>
<div class="container hidden" id="dashboardTab"><div class="panel"><h2>Dashboard Management</h2><p class="small">Placeholder</p></div></div>
<div class="container hidden" id="wifiTab"><div class="panel"><h2>WiFi Management</h2><p class="small">Placeholder</p></div></div>

<script>
/* SPA + polling */
function showTab(tab){
  const ids=['home','analytics','inventory','offers','dashboard','wifi'];
  ids.forEach(id=>{
    const el=document.getElementById(id+'Tab'); if(!el) return;
    if(id===tab) el.classList.remove('hidden'); else el.classList.add('hidden');
  });
  document.querySelectorAll('.tabs a').forEach(a=>a.classList.remove('active'));
  const map = {home:0,analytics:1,inventory:2,offers:3,dashboard:4,wifi:5};
  const idx = map[tab] || 0; const links = document.querySelectorAll('.tabs a');
  if(links[idx]) links[idx].classList.add('active');
  if(tab==='analytics') renderPopularTimes();
}

async function refreshUI(){
  try{
    const res = await fetch('/current');
    const arr = await res.json();
    const cc = document.getElementById('current_cards');
    cc.innerHTML='';
    if(!arr || arr.length===0){
      cc.innerHTML='<div class="small">No current detections.</div>';
    }
    // use detection id to fetch matching image (fixes mismatch)
    (arr||[]).slice(0,12).forEach((d,i)=>{
      const div = document.createElement('div');
      div.className='card';
      let imgTag = d.has_img ? `<img src="/current_img_id/${d.id}" class="face" onerror="this.src='/static/faces/placeholder.png'"/>` : '';
      div.innerHTML = `<div class="small">${d.zone||''}</div><div><b>${d.name||'Unknown'}</b></div><div class="small">Gender: ${d.gender||''} ${d.age?(' | '+d.age):''}</div><div class="small">Cats: ${d.categories||''}</div>${imgTag}`;
      cc.appendChild(div);
    });

    const r2 = await fetch('/alerts');
    const allAlerts = await r2.json();
    const as = document.getElementById('alerts');
    as.innerHTML='';
    // show only latest 10 alerts
    (allAlerts||[]).slice(0,10).forEach(a=>{
      const cls = (a.alert_type==='fire'||a.alert_type==='weapon_threat')?'alert-item red':((a.alert_type==='smoke'||a.alert_type==='queue_overflow'||a.alert_type==='crowd')?'alert-item warn':'alert-item');
      const el = document.createElement('div');
      el.className=cls;
      el.innerHTML = `<div class="small">${a.timestamp||''}</div><div><b>${a.alert_type||''}</b> — ${a.info||''}</div>`;
      as.appendChild(el);
    });
  }catch(e){
    console.error(e);
  }
}

setInterval(()=>{
  refreshUI().catch(()=>{});
  if(!document.getElementById('analyticsTab').classList.contains('hidden')) renderPopularTimes().catch(()=>{});
}, 12000);

document.addEventListener('DOMContentLoaded', ()=>{
  showTab('home');
  refreshUI();
});

/* Popular Times vertical chart (Plotly) */
async function fetchAnalytics(){ const r = await fetch('/analytics_data'); return await r.json(); }

function hourLabels(){
  const hrs=[];
  for(let h=9; h<=21; h++){ hrs.push((h<10?('0'+h):h)+':00'); }
  return hrs;
}

async function renderPopularTimes(){
  try{
    const data = await fetchAnalytics();
    const days = data.days || [];
    const hours = data.hours && data.hours.length ? data.hours : hourLabels();
    const sel = document.getElementById('day_select');
    const prev = sel.value;
    sel.innerHTML = '';
    days.forEach(d=>{
      const o=document.createElement('option');
      o.value=d; o.text=d; sel.appendChild(o);
    });
    const selected = (prev && days.includes(prev)) ? prev : (days.length?days[days.length-1]:null);
    if(selected) sel.value = selected;

    const hist = (selected && data.visitors_per_day && data.visitors_per_day[selected]) ? data.visitors_per_day[selected] : hours.map(()=>0);

    // determine peak and current hour usage
    const peak = hist.length ? Math.max(...hist) : 0;
    const now = new Date();
    const currentHour = (now.getHours()<10?('0'+now.getHours()):now.getHours()) + ':00';
    const currentIdx = hours.indexOf(currentHour);
    const currentVisitors = (currentIdx >= 0 && currentIdx < hist.length) ? hist[currentIdx] : 0;

    // status heuristic based on current hour vs peak
    let status = 'Not too busy';
    if(peak > 0){
      if(currentVisitors < peak * 0.25) status = 'Not too busy';
      else if(currentVisitors < peak * 0.6) status = 'Moderate';
      else status = 'Busy';
    } else {
      status = 'Not too busy';
    }

    // update subtitle (LIVE badge is separate in markup)
    const subtitle = document.getElementById('pt-sub');
    if(subtitle){
      subtitle.innerText = `${status} — peak ${peak} visitors`;
    }

    // colors: base blue, highlight current hour red (if current hour has value)
    const baseColor = '#7fb3ff';
    const activeColor = '#ff4d6d';
    const colors = hist.map((v,i)=> (i===currentIdx && v>0) ? activeColor : baseColor);

    // load plotly if not present
    if(typeof Plotly === 'undefined'){
      const s=document.createElement('script');
      s.src='https://cdn.plot.ly/plotly-latest.min.js';
      document.head.appendChild(s);
      s.onload = ()=> renderPopularTimes();
      return;
    }

    // vertical bar trace
    const trace = {
      x: hours, y: hist, type: 'bar',
      marker: { color: colors, line: { width: 0 } },
      hovertemplate: '%{x}<br>%{y} visitors<extra></extra>',
      text: hist.map(String), textposition: 'outside'
    };
    const layout = {
      paper_bgcolor:'#0b1220', plot_bgcolor:'#0b1220',
      xaxis: { title: 'Hour of Day', tickfont:{color:'#e6edf3'}, tickangle:0, showgrid:false },
      yaxis: { title: 'Visitors', tickfont:{color:'#e6edf3'}, showgrid:true, gridcolor:'#17202a' },
      margin: {l:60, r:20, t:40, b:60}, font: { color:'#e6edf3' }, showlegend:false, bargap: 0.2,
      title: { text: 'Popular Times — visitors per hour', font:{ color:'#e6edf3' } }
    };
    Plotly.react('popular_times_div', [trace], layout, {displayModeBar:false});

    // Visitors by Zone
    const zoneObj = data.zones_per_day && data.zones_per_day[selected] ? data.zones_per_day[selected] : {};
    const zoneKeys = Object.keys(zoneObj);
    const zoneVals = zoneKeys.map(k => zoneObj[k]||0);
    Plotly.react('chart_zone', [{x:zoneKeys, y:zoneVals, type:'bar', marker:{color:'#f59e0b'}}], {paper_bgcolor:'#0b1220', plot_bgcolor:'#0b1220', font:{color:'#e6edf3'}, margin:{l:40}}, {displayModeBar:false});

    // Visitors by Gender/Age
    const genderObj = data.gender_counts_per_day && data.gender_counts_per_day[selected] ? data.gender_counts_per_day[selected] : {};
    const ageObj = data.age_buckets_per_day && data.age_buckets_per_day[selected] ? data.age_buckets_per_day[selected] : null;
    if(ageObj && data.age_gender_map && data.age_gender_map[selected]){
      // age buckets by gender available -> grouped bar
      const ageLabels = data.age_gender_map[selected].age_labels || Object.keys(ageObj);
      const genders = Object.keys(data.age_gender_map[selected]).filter(k=>k!=='age_labels');
      const traces = genders.map(g=>{
        return { x: ageLabels, y: ageLabels.map(a => (data.age_gender_map[selected][g] && data.age_gender_map[selected][g][a]) ? data.age_gender_map[selected][g][a] : 0), name: g, type: 'bar' };
      });
      Plotly.react('chart_gender', traces, {barmode:'group', paper_bgcolor:'#0b1220', plot_bgcolor:'#0b1220', font:{color:'#e6edf3'}, margin:{l:40}}, {displayModeBar:false});
    } else {
      // fallback: pie chart of genders
      const gKeys = Object.keys(genderObj);
      const gVals = gKeys.map(k=>genderObj[k]||0);
      Plotly.react('chart_gender', [{labels:gKeys, values:gVals, type:'pie', marker:{}}], {paper_bgcolor:'#0b1220', plot_bgcolor:'#0b1220', font:{color:'#e6edf3'}}, {displayModeBar:false});
    }

  }catch(e){
    console.error('renderPopularTimes error', e);
  }
}
</script>
</body>
</html>
"""

# ---------- Flask endpoints ----------
app = Flask(__name__, static_folder="static")

# serve SPA on common routes to avoid 404 on direct navigation
@app.route("/")
@app.route("/analytics")
@app.route("/inventory")
@app.route("/offers")
@app.route("/dashboard")
@app.route("/wifi")
def index():
    return render_template_string(HTML)

@app.route("/video1")
def video1():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/current")
def current():
    out=[]
    # return stable id for each detection so image -> data mapping remains correct
    for i, d in enumerate(last_detections[:16]):
        # stable-ish short id for this detection object while it exists in memory
        img_id = hashlib.md5(str(id(d)).encode()).hexdigest()[:8]
        out.append({
            "id": img_id,
            "name": d.get("name",""),
            "categories": ";".join(d.get("categories") or []),
            "zone": d.get("zone",""),
            "gender": d.get("gender",""),
            "age": d.get("age",""),
            "has_img": True if d.get("img") is not None else False,
            "has_face": True if d.get("face") is not None else False
        })
    return jsonify(out)

@app.route("/current_img/<int:idx>")
def current_img(idx):
    # keep old endpoint (index-based) for backward compatibility
    if idx < 0 or idx >= len(last_detections):
        return make_response("", 404)
    img = last_detections[idx].get("img")
    if img is None:
        return make_response("", 404)
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return make_response("", 500)
    return make_response(buf.tobytes(), 200, {"Content-Type":"image/jpeg"})

@app.route("/current_img_id/<string:img_id>")
def current_img_id(img_id):
    # new: return image by stable id generated in /current
    for d in last_detections:
        try:
            if hashlib.md5(str(id(d)).encode()).hexdigest()[:8] == img_id:
                img = d.get("img")
                if img is None:
                    return make_response("", 404)
                ok, buf = cv2.imencode(".jpg", img)
                if not ok:
                    return make_response("", 500)
                return make_response(buf.tobytes(), 200, {"Content-Type":"image/jpeg"})
        except Exception:
            continue
    return make_response("", 404)

@app.route("/alerts")
def alerts_api():
    combined = []
    # read alert rows from CSV
    try:
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
            if "alert" in df.columns:
                df2 = df[df["alert"].notna() & (df["alert"].astype(str) != "")]
                df2 = df2.sort_values("timestamp", ascending=False).head(200)
                for _, r in df2.iterrows():
                    combined.append({
                        "timestamp": str(r.get("timestamp") or ""),
                        "alert_type": str(r.get("alert") or ""),
                        "info": f"name={r.get('name','')} zone={r.get('zone','')}"
                    })
    except Exception as e:
        print("alerts_api read error:", e)

    # prepend in-memory alerts (limit)
    for a in alert_service.alerts[:10]:
        combined.insert(0, a)

    # dedupe while preserving order
    seen = set(); out = []
    for c in combined:
        key = (c.get("timestamp",""), c.get("alert_type",""), c.get("info",""))
        if key in seen: continue
        seen.add(key); out.append(c)

    # return only latest 10 alerts to the UI
    return jsonify(out[:10])

@app.route("/upload_image", methods=["POST"])
def upload_image():
    try:
        file = request.files.get("image")
        name = (request.form.get("name") or "").strip()
        categories = (request.form.get("categories") or "").strip()
        if not file or not name:
            return jsonify({"ok":False, "msg":"invalid"}), 400
        fn = secure_filename(file.filename)
        out_fn = f"{uuid.uuid4().hex}_{fn}"
        out_path = os.path.join(KNOWN_IMAGES_DIR, out_fn)
        file.save(out_path)
        df = pd.read_csv(FACES_CSV) if os.path.exists(FACES_CSV) else pd.DataFrame(columns=["filename","name","categories"])
        df.loc[len(df)] = {"filename": out_fn, "name": name, "categories": categories}
        df.to_csv(FACES_CSV, index=False)
        global known_encodings, known_meta
        known_encodings, known_meta = load_known_faces()
        return redirect(url_for("index"))
    except Exception as e:
        return jsonify({"ok":False, "msg":str(e)}), 500

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"ok":False, "msg":"no file"}), 400
        df = pd.read_csv(file)
        for col in ["filename","name","categories"]:
            if col not in df.columns:
                return jsonify({"ok":False, "msg":"missing columns"}), 400
        df.to_csv(FACES_CSV, index=False)
        global known_encodings, known_meta
        known_encodings, known_meta = load_known_faces()
        return redirect(url_for("index"))
    except Exception as e:
        return jsonify({"ok":False, "msg":str(e)}), 500

@app.route("/download_faces_csv")
def download_faces_csv():
    ensure_faces_csv()
    return send_file(FACES_CSV, as_attachment=True, download_name="faces.csv")

def _build_analytics_from_sample(sample_df):
    """ sample_df expected to contain columns: date (YYYY-MM-DD), hour (HH:00), visitors (int), wait_time (optional numeric), zone (optional), gender (optional), age_bucket (optional)
    We'll aggregate per-day and per-hour.
    """
    sample_df = sample_df.copy()
    if 'date' not in sample_df.columns:
        sample_df['date'] = datetime.now().date().isoformat()
    if 'hour' not in sample_df.columns and 'timestamp' in sample_df.columns:
        sample_df['hour'] = pd.to_datetime(sample_df['timestamp'], errors='coerce').dt.strftime('%H:00')
    if 'visitors' not in sample_df.columns:
        # try to infer from single detections: count rows per hour
        sample_df['visitors'] = 1
    # ensure hour is in HH:00 format
    sample_df['hour'] = sample_df['hour'].astype(str).str.slice(0,5)
    days = sorted(sample_df['date'].dropna().unique())
    hours = [f"{str(h).zfill(2)}:00" for h in range(9,22)]
    visitors_per_day = {}
    avg_wait_per_day = {}
    zones_per_day = {}
    gender_counts_per_day = {}
    age_buckets_per_day = {}
    age_gender_map = {}
    for d in days:
        day_df = sample_df[sample_df['date']==d]
        visitors_per_day[d] = [int(day_df[day_df['hour']==h]['visitors'].sum()) for h in hours]
        # avg wait per hour if provided (analytics endpoint may return avg_wait_per_hour)
        if 'wait_time' in day_df.columns:
            avg_wait_per_day[d] = [float(day_df[day_df['hour']==h]['wait_time'].mean() or 0) for h in hours]
        else:
            avg_wait_per_day[d] = [0 for _ in hours]
        # zones
        if 'zone' in day_df.columns:
            zcounts = day_df['zone'].fillna('').replace('', 'Cloth Section')
            zones_per_day[d] = zcounts.value_counts().to_dict()
        else:
            zones_per_day[d] = {}
        # gender
        if 'gender' in day_df.columns:
            gender_counts_per_day[d] = day_df['gender'].fillna('Unknown').replace('', 'Unknown').value_counts().to_dict()
        else:
            gender_counts_per_day[d] = {}
        # age buckets and age x gender map
        if 'age_bucket' in day_df.columns and 'gender' in day_df.columns:
            ag = day_df.groupby(['gender','age_bucket']).size().unstack(fill_value=0)
            age_buckets_per_day[d] = ag.sum(axis=0).to_dict()
            age_gender_map[d] = {'age_labels': list(ag.columns)}
            for g in ag.index:
                age_gender_map[d][g] = ag.loc[g].to_dict()
        else:
            age_buckets_per_day[d] = {}
            age_gender_map[d] = None
    return {
        "days": days,
        "hours": hours,
        "visitors_per_day": visitors_per_day,
        "avg_wait_per_hour": avg_wait_per_day,
        "zones_per_day": zones_per_day,
        "gender_counts_per_day": gender_counts_per_day,
        "age_buckets_per_day": age_buckets_per_day,
        "age_gender_map": age_gender_map
    }

@app.route("/analytics_data")
def analytics_data():
    """ Return analytics structure used by the frontend. Primary source: LOG_FILE (detections_log.csv) where each detection writes a timestamp and zone/gender/age_bucket. Fallback (manual): SAMPLE_FILE (popular_times_sample.csv) if present and LOG_FILE doesn't have enough data. """
    try:
        # if LOG_FILE exists and contains timestamped detections, prefer it
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE)>0:
            df = pd.read_csv(LOG_FILE)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df["date"] = df["timestamp"].dt.date.astype(str)
                df["hour"] = df["timestamp"].dt.strftime("%H:00")
            else:
                df["date"] = ""; df["hour"] = ""
            hours = [f"{str(h).zfill(2)}:00" for h in range(9,22)]
            days = sorted(df["date"].dropna().unique())
            visitors_per_day = {}; gender_counts_per_day = {}; zones_per_day = {}; age_buckets_per_day = {}; age_gender_map = {}; avg_wait_per_hour = {}
            for d in days:
                day_df = df[df["date"]==d]
                # visitors per hour is simply count per hour
                visitors_per_day[d] = [int(day_df[day_df["hour"]==h].shape[0]) for h in hours]
                # zones
                zcounts = day_df["zone"].fillna("").replace("", "Cloth Section").apply(lambda x: (x if x!='unknown' else 'Cloth Section'))
                zones_per_day[d] = zcounts.value_counts().to_dict()
                # gender counts
                gender_counts_per_day[d] = day_df["gender"].fillna("Unknown").replace("", "Unknown").value_counts().to_dict()
                # age buckets distribution
                if "age_bucket" in day_df.columns:
                    age_buckets_per_day[d] = day_df["age_bucket"].fillna("Unknown").replace("", "Unknown").value_counts().to_dict()
                else:
                    age_buckets_per_day[d] = {}
                # age x gender map if both available
                if "age_bucket" in day_df.columns and "gender" in day_df.columns:
                    ag = day_df.groupby(['gender','age_bucket']).size().unstack(fill_value=0)
                    age_gender_map[d] = {'age_labels': list(ag.columns)}
                    for g in ag.index:
                        age_gender_map[d][g] = ag.loc[g].to_dict()
                else:
                    age_gender_map[d] = None
                # avg wait per hour: if log contains 'wait_time' column (rare), use it; else zeros
                if "wait_time" in day_df.columns:
                    avg_wait_per_hour[d] = [float(day_df[day_df["hour"]==h]["wait_time"].mean() or 0) for h in hours]
                else:
                    avg_wait_per_hour[d] = [0 for _ in hours]
            return jsonify({
                "days": list(days),
                "hours": hours,
                "visitors_per_day": visitors_per_day,
                "gender_counts_per_day": gender_counts_per_day,
                "zones_per_day": zones_per_day,
                "age_buckets_per_day": age_buckets_per_day,
                "age_gender_map": age_gender_map,
                "avg_wait_per_hour": avg_wait_per_hour
            })
        else:
            # fallback: if SAMPLE_FILE exists (manual), load it to synthesize analytics
            if os.path.exists(SAMPLE_FILE):
                sample_df = pd.read_csv(SAMPLE_FILE)
                out = _build_analytics_from_sample(sample_df)
                return jsonify(out)
        # no data available
        return jsonify({"days":[], "hours":[], "visitors_per_day":{}, "gender_counts_per_day":{}, "zones_per_day":{}, "age_buckets_per_day":{}, "age_gender_map":{}, "avg_wait_per_hour":{}})
    except Exception as e:
        print("analytics_data error:", e)
        return jsonify({"days":[], "hours":[], "visitors_per_day":{}, "gender_counts_per_day":{}, "zones_per_day":{}, "age_buckets_per_day":{}, "age_gender_map":{}, "avg_wait_per_hour":{}})

# ---------- start ----------
if __name__ == "__main__":
    print("Starting ipcam_yolo4_final at http://0.0.0.0:5000")
    # placeholder image for faces
    placeholder = os.path.join("static","faces","placeholder.png")
    os.makedirs(os.path.dirname(placeholder), exist_ok=True)
    if not os.path.exists(placeholder) and cv2 is not None:
        try:
            p = 255*np.ones((120,120,3), dtype=np.uint8)
            cv2.putText(p, "No Image", (6,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            cv2.imwrite(placeholder, p)
        except Exception:
            pass
    app.run(host="0.0.0.0", port=5000, threaded=True)
