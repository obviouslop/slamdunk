#!/usr/bin/env python3
"""Stereo viewer for a still.tiff (tools/stills.py): writes ONE self-contained HTML file (no server, no external libraries)
and optionally opens it in the browser.

  stillview.py captures/stills/<name>/still.tiff [--open] [--max 1600] [--out FILE]

Views: rectified stereo pair as red-cyan ANAGLYPH (with convergence shift), SIDE-BY-SIDE parallel or cross-eyed, BLINK (flicker
left/right to see the parallax), LEFT / RIGHT full-resolution upright camera images, DEPTH map (turbo colours, overlay opacity on
the rectified left image) and the composited headset view. Click in a rectified view (anaglyph / pair / blink / depth) to read the
stereo depth and the 3-D point (camera + STAGE coordinates); click two points to measure the distance between them.
Keys: 1-7 views, B blink, M measure on/off, C clear, arrows shift the convergence, +/- zoom.
"""
import argparse, base64, json, os, sys, webbrowser
import numpy as np
import cv2
import tifffile


def load(path):
    out = {}
    with tifffile.TiffFile(path) as t:
        for pg in t.pages:
            d = json.loads(pg.description) if pg.description.startswith("{") else {}
            out[d.get("role", pg.tags["PageName"].value if "PageName" in pg.tags else str(pg.index))] = (d, pg.asarray())
    return out


def jpg64(rgb, maxd, q=88):
    h, w = rgb.shape[:2]
    if max(h, w) > maxd:
        s = maxd / max(h, w); rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", np.ascontiguousarray(rgb[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()).decode()


def build(path, maxd=1600):
    P = load(path)
    need = [r for r in ("left", "right", "rect_left", "rect_right", "depth") if r not in P]
    if need: sys.exit(f"{path}: missing pages {need} (old-format still?)")
    meta = P["left"][0].get("metadata", {})
    dd, Z = P["depth"]
    geom = {k: dd[k] for k in ("intrinsics_rect", "stage_from_rect", "baseline_m", "k", "d0_px") if k in dd}
    imgs = {"left": jpg64(P["left"][1], maxd), "right": jpg64(P["right"][1], maxd), "rl": jpg64(P["rect_left"][1], 1232), "rr": jpg64(P["rect_right"][1], 1232)}
    if "composited" in P: imgs["comp"] = jpg64(P["composited"][1], 1920)
    z16 = np.clip(np.round(Z * 1000), 0, 65535).astype("<u2")             # millimetres, 0 = invalid
    data = dict(imgs=imgs, depth=base64.b64encode(z16.tobytes()).decode(), dw=int(Z.shape[1]), dh=int(Z.shape[0]), geom=geom,
                meta={k: v for k, v in meta.items() if k not in ("calibration", "geometry", "tiff_pages", "stage_from_cam", "stage_from_cam_jpg")},
                name=os.path.basename(os.path.dirname(os.path.abspath(path))), comp=P.get("composited", ({},))[0])
    return HTML.replace("/*DATA*/", "const D = " + json.dumps(data) + ";")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Still viewer</title><style>
:root{--bg:#101418;--fg:#dfe6ee;--mut:#8b98a8;--acc:#4cc2ff;--pan:#1a2027}
@media (prefers-color-scheme: light){:root{--bg:#f3f5f7;--fg:#1b2430;--mut:#5d6b7c;--acc:#0a6fb8;--pan:#e4e9ee}}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
#bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:8px 12px;background:var(--pan)}
button,label{font:inherit}button{background:transparent;color:var(--fg);border:1px solid var(--mut);border-radius:6px;padding:4px 10px;cursor:pointer}
button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
#wrap{display:flex;height:calc(100% - 52px)}#view{flex:1;overflow:auto;position:relative}
canvas{display:block;image-rendering:auto;cursor:crosshair}
#side{width:340px;overflow:auto;padding:10px 12px;background:var(--pan);font-size:12.5px}
pre{white-space:pre-wrap;word-break:break-word;margin:4px 0;color:var(--mut)}h4{margin:10px 0 4px}
#read{position:absolute;left:10px;top:10px;background:#000a;color:#fff;padding:6px 9px;border-radius:6px;font-size:12.5px;pointer-events:none;white-space:pre}
input[type=range]{vertical-align:middle;width:110px}@media(max-width:800px){#side{display:none}}
</style></head><body>
<div id="bar"><b id="title"></b>
<button data-v="anaglyph" class="on">1 Anaglyph</button><button data-v="sbs">2 Side-by-side</button><button data-v="blink">3 Blink</button>
<button data-v="left">4 Left</button><button data-v="right">5 Right</button><button data-v="depth">6 Depth</button><button data-v="comp">7 Composited</button>
<label><input type="checkbox" id="cross"> cross-eyed</label>
<label>shift <input type="range" id="shift" min="-60" max="60" value="0"> <span id="shv">0</span> px</label>
<label>depth overlay <input type="range" id="opac" min="0" max="100" value="55"></label>
<label>zoom <input type="range" id="zoom" min="25" max="300" value="100"></label>
<button id="meas">M Measure</button></div>
<div id="wrap"><div id="view"><canvas id="cv"></canvas><div id="read">click: depth / 3-D point</div></div><div id="side"><h4>Capture</h4><pre id="info"></pre><h4>Metadata</h4><pre id="meta"></pre></div></div>
<script>
/*DATA*/
const $=id=>document.getElementById(id), cv=$("cv"), cx=cv.getContext("2d",{willReadFrequently:true});
const I={}; let view="anaglyph", measure=false, pts=[], blinkOn=true, timer=null;
const G=D.geom, f=G.intrinsics_rect.f, cx0=G.intrinsics_rect.cx, cy0=G.intrinsics_rect.cy, S=G.stage_from_rect, DW=D.dw, DH=D.dh;
const raw=Uint8Array.from(atob(D.depth),c=>c.charCodeAt(0)); const Zmm=new Uint16Array(raw.buffer);
$("title").textContent=D.name; $("meta").textContent=JSON.stringify(D.meta,null,1);
const m=D.meta; $("info").textContent=[`time  ${m.t_utc_iso||""}`,`frame ${m.left?m.left.frame:""}  ts ${m.left?m.left.ts:""}`,`exposure ${m.left?m.left.exp_lines:""} lines, gain ${m.left?m.left.gain_code:""}`,
 `baseline ${(G.baseline_m*1000).toFixed(1)} mm   f ${f.toFixed(1)} px   depth k ${G.k} d0 ${G.d0_px}`,`composited: ${D.comp&&D.comp.source?D.comp.source+" (+"+D.comp.after_trigger_s+" s)":"none"}`].join("\n");
let loaded=0; const names=Object.keys(D.imgs);
names.forEach(k=>{const im=new Image(); im.onload=()=>{I[k]=im; if(++loaded==names.length) draw();}; im.src=D.imgs[k];});
function turbo(t){t=Math.min(1,Math.max(0,t));const r=Math.round(255*Math.min(1,Math.max(0,1.5-Math.abs(4*t-3)+0.1*t))),g=Math.round(255*Math.min(1,Math.max(0,1.5-Math.abs(4*t-2)))),b=Math.round(255*Math.min(1,Math.max(0,1.5-Math.abs(4*t-1))));return[r,g,b];}
function fit(img,w,h){cv.width=w;cv.height=h;cv.style.width=w*$("zoom").value/100+"px";cv.style.height=h*$("zoom").value/100+"px";}
function draw(){
 const rl=I.rl, rr=I.rr; if(!rl) return; const sh=+$("shift").value; $("shv").textContent=sh;
 if(view=="anaglyph"){ fit(rl,rl.width,rl.height);
  const a=document.createElement("canvas"); a.width=rl.width;a.height=rl.height; const ca=a.getContext("2d"); ca.drawImage(rl,0,0); const A=ca.getImageData(0,0,a.width,a.height);
  ca.clearRect(0,0,a.width,a.height); ca.drawImage(rr,sh,0); const B=ca.getImageData(0,0,a.width,a.height); const o=A.data,p=B.data;
  for(let i=0;i<o.length;i+=4){const l=0.299*o[i]+0.587*o[i+1]+0.114*o[i+2], r=0.299*p[i]+0.587*p[i+1]+0.114*p[i+2]; o[i]=l; o[i+1]=r; o[i+2]=r;}
  cx.putImageData(A,0,0);
 } else if(view=="sbs"){ const w=rl.width; fit(rl,2*w+10,rl.height); cx.fillStyle="#000";cx.fillRect(0,0,cv.width,cv.height);
  const [a,b]=$("cross").checked?[rr,rl]:[rl,rr]; cx.drawImage(a,0,0); cx.drawImage(b,w+10+sh,0);
 } else if(view=="blink"){ fit(rl,rl.width,rl.height); cx.clearRect(0,0,cv.width,cv.height); if(blinkOn) cx.drawImage(rl,0,0); else cx.drawImage(rr,sh,0);
 } else if(view=="left"){ fit(I.left,I.left.width,I.left.height); cx.drawImage(I.left,0,0);
 } else if(view=="right"){ fit(I.right,I.right.width,I.right.height); cx.drawImage(I.right,0,0);
 } else if(view=="comp"){ if(!I.comp){fit(rl,400,100);cx.fillText("no composited frame in this still",10,50);return;} fit(I.comp,I.comp.width,I.comp.height); cx.drawImage(I.comp,0,0);
 } else if(view=="depth"){ fit(rl,rl.width,rl.height); cx.drawImage(rl,0,0); const A=cx.getImageData(0,0,rl.width,rl.height), op=$("opac").value/100, sc=rl.width/DW;
  for(let y=0;y<rl.height;y++)for(let x=0;x<rl.width;x++){const z=Zmm[Math.min(DH-1,(y/sc)|0)*DW+Math.min(DW-1,(x/sc)|0)]; if(!z)continue; const c=turbo(1-(z/1000-0.3)/3.2), i=(y*rl.width+x)*4;
   A.data[i]=A.data[i]*(1-op)+c[0]*op;A.data[i+1]=A.data[i+1]*(1-op)+c[1]*op;A.data[i+2]=A.data[i+2]*(1-op)+c[2]*op;}
  cx.putImageData(A,0,0); }
 // measurement markers
 if(["anaglyph","sbs","blink","depth"].includes(view)){ cx.strokeStyle="#ff0";cx.fillStyle="#ff0";cx.lineWidth=2;
  pts.forEach((p,i)=>{cx.beginPath();cx.arc(p.u,p.v,6,0,7);cx.stroke(); if(i&&pts[i-1]){cx.beginPath();cx.moveTo(pts[i-1].u,pts[i-1].v);cx.lineTo(p.u,p.v);cx.stroke();}});}
}
function depthAt(u,v){const sc=I.rl.width/DW, x=Math.min(DW-1,Math.max(0,(u/sc)|0)), y=Math.min(DH-1,Math.max(0,(v/sc)|0)); let z=Zmm[y*DW+x];
 if(!z){for(let r=1;r<5&&!z;r++)for(let dy=-r;dy<=r&&!z;dy++)for(let dx=-r;dx<=r;dx++){const xx=x+dx,yy=y+dy; if(xx>=0&&yy>=0&&xx<DW&&yy<DH&&Zmm[yy*DW+xx]){z=Zmm[yy*DW+xx];break;}}} return z/1000;}
function point(u,v){const z=depthAt(u,v); if(!z) return null; const sc=I.rl.width/DW, X=((u-cx0)/f)*z, Y=((v-cy0)/f)*z; const c=[X,Y,z];
 const s=[0,1,2].map(r=>S[r][0]*X+S[r][1]*Y+S[r][2]*z+S[r][3]); return {u,v,z,c,s};}
cv.addEventListener("click",e=>{ if(!["anaglyph","sbs","blink","depth"].includes(view)) return;
 const r=cv.getBoundingClientRect(), k=cv.width/r.width; let u=(e.clientX-r.left)*k, v=(e.clientY-r.top)*k; if(view=="sbs"){u=u%(I.rl.width+10);}
 const p=point(u,v); if(!p){$("read").textContent="no depth here";return;}
 let t=`depth ${p.z.toFixed(3)} m\ncamera  ${p.c.map(x=>x.toFixed(3)).join(", ")}\nSTAGE   ${p.s.map(x=>x.toFixed(3)).join(", ")}`;
 if(measure){ if(pts.length>=2) pts=[]; pts.push(p); if(pts.length==2){const d=Math.hypot(...[0,1,2].map(i=>pts[0].s[i]-pts[1].s[i])); t+=`\n\ndistance ${(d*100).toFixed(1)} cm`;} } else pts=[p];
 $("read").textContent=t; draw();});
function setView(v){view=v; document.querySelectorAll("#bar button[data-v]").forEach(b=>b.classList.toggle("on",b.dataset.v==v)); clearInterval(timer); if(v=="blink") timer=setInterval(()=>{blinkOn=!blinkOn;draw();},450); draw();}
document.querySelectorAll("#bar button[data-v]").forEach(b=>b.onclick=()=>setView(b.dataset.v));
["shift","opac","zoom","cross"].forEach(i=>$(i).oninput=draw);
$("meas").onclick=()=>{measure=!measure;$("meas").classList.toggle("on",measure);pts=[];draw();};
addEventListener("keydown",e=>{const k=e.key; if("1234567".includes(k)&&k.length==1) setView(["anaglyph","sbs","blink","left","right","depth","comp"][+k-1]);
 else if(k=="b"||k=="B") setView("blink"); else if(k=="m"||k=="M") $("meas").click(); else if(k=="c"||k=="C"){pts=[];draw();}
 else if(k=="ArrowLeft"){$("shift").value=+$("shift").value-1;draw();} else if(k=="ArrowRight"){$("shift").value=+$("shift").value+1;draw();}
 else if(k=="+"||k=="="){$("zoom").value=Math.min(300,+$("zoom").value+10);draw();} else if(k=="-"){$("zoom").value=Math.max(25,+$("zoom").value-10);draw();}});
</script></body></html>"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("tiff"); ap.add_argument("--open", action="store_true")
    ap.add_argument("--max", type=int, default=1600, help="longest side of the embedded upright left/right images"); ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.tiff)), "still_view.html")
    open(out, "w").write(build(a.tiff, a.max)); print(out, f"({os.path.getsize(out) / 1e6:.1f} MB)")
    if a.open: webbrowser.open("file://" + os.path.abspath(out))
