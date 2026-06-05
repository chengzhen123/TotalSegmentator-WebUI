import gc
import json
import shutil
from pathlib import Path
import os, glob, time, io
import traceback
from threading import Thread
import subprocess
import datetime
import zipfile

import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory, after_this_request
from markupsafe import escape
from flask_cors import CORS

STORE_DIR = Path("store")
STORE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = STORE_DIR / "uploaded_images"
UPLOAD_DIR.mkdir(exist_ok=True)
TMP_PREVIEW = STORE_DIR / "tmp_preview"
TMP_PREVIEW.mkdir(exist_ok=True)

def log(text, p=False):
    if p:
        print(text)
    with open(STORE_DIR / "log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now()} {text}\n")

def has_valid_credentials(api_key):
    key_file = STORE_DIR / "api_keys.txt"
    if not key_file.exists():
        return False
    with open(key_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or ":" not in line:
            continue
        _, key = line.split(":", maxsplit=2)
        if key.strip() == api_key.strip():
            return True
    return False

app = Flask(__name__)
# 【微调】CORS 显式暴露 X-Task-Id，允许前端 JavaScript 读取自定义 Header
CORS(app, expose_headers=["X-Task-Id"])

@app.route("/preview/<task_id>/<fname>", methods=["GET"])
def get_preview_file(task_id, fname):
    task_path = TMP_PREVIEW / task_id
    return send_from_directory(task_path, fname)

# 托管本地 0.69.0 版本的 niivue.umd.js 文件
@app.route("/static/niivue.umd.js", methods=["GET"])
def get_local_niivue():
    current_dir = Path(__file__).parent
    return send_from_directory(current_dir, "niivue.umd.js", mimetype="application/javascript")


@app.route("/", methods=["GET"])
def index_html():
    html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>TotalSegmentator 分割预览 (0.69.0 UMD 完美修复版)</title>
    <style>
        * {box-sizing: border-box;margin:0;padding:0;font-family:system-ui;}
        .container{max-width:1400px;margin:20px auto;padding:20px;border:1px solid #ddd;border-radius:10px;}
        .main-row{display:flex;gap:20px;flex-wrap:wrap;}
        .col-left{flex:1;min-width:320px}
        .col-right{flex:2.5;min-width:700px}
        .view-row{display:flex;gap:10px;margin-bottom:20px;}
        .view-item{flex:1;border:2px dashed #999;padding:8px;border-radius:6px;background:#f9f9f9;height:420px;position:relative;}
        .png-wrap{width:100%;border:1px solid #ccc;padding:10px;border-radius:6px;background:#f9f9f9;margin-top:10px;}
        #pngImg{max-width:100%;height:auto;display:block;margin:0 auto;}
        h2{text-align:center;margin-bottom:20px;}
        h4{text-align:center;margin:6px 0;color:#333;font-size:14px}
        .item{margin-bottom:18px}
        label{display:block;margin-bottom:8px;font-weight:500}
        input[type="text"],input[type="file"]{width:100%;padding:10px;border:1px solid #ccc;border-radius:6px;}
        .check-item{display:flex;align-items:center;gap:8px}
        button{width:100%;padding:12px;border:none;border-radius:6px;background:#2563eb;color:#fff;font-size:16px;cursor:pointer;}
        button:disabled{background:#87a5e0;cursor:not-allowed;}
        #status-text{margin:12px 0;padding:8px;border-radius:4px;min-height:36px;}
        .success{background:#dcfce7;color:#166534}
        .error{background:#fee2e2;color:#991b1b}
        .loading{background:#eff6ff;color:#1e40af}
        .tip{font-size:13px;color:#666;margin:6px 0}
        canvas{width:100%;height:calc(100% - 30px);background:#000;display:block;}
    </style>
    <script src="/static/niivue.umd.js"></script>
</head>
<body>
<div class="container">
    <h2>TotalSegmentator AI器官分割 for CT&MRI</h2>
    <div class="main-row">
        <div class="col-left">
            <div class="item">
                <label>密钥</label>
                <input id="apiKey" value="123456">
            </div>
            <div class="item">
                <label>选择nii.gz文件</label>
                <input type="file" id="fileInput" accept=".nii,.nii.gz">
            </div>
            <div class="item check-item">
                <input type="checkbox" id="openStats">
                <label>开启体积统计</label>
            </div>
            <div id="status-text"></div>
            <div class="item"><button id="checkBtn">检测服务状态</button></div>
            <div class="item"><button id="runBtn">上传&GPU分割</button></div>
        </div>
        <div class="col-right">
            <div class="view-row">
                <div class="view-item"><h4>原始 CT/MRI (source.nii.gz)</h4><canvas id="c1"></canvas></div>
                <div class="view-item"><h4>AI 分割标签 (seg.nii.gz)</h4><canvas id="c2"></canvas></div>
            </div>
            <div class="png-wrap">
                <h4>全景效果图 preview_total.png</h4>
                <img id="pngImg">
            </div>
            <div class="tip">提示：在上方黑框内按住鼠标左键拖拽可以调整窗宽窗位，滚动鼠标滚轮可以切换切片。</div>
        </div>
    </div>
</div>

<script>
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => e.preventDefault());

const $id = s => document.getElementById(s);
const apiKey = $id('apiKey');
const fileInp = $id('fileInput');
const ckStat = $id('openStats');
const statusBox = $id('status-text');
const checkBtn = $id('checkBtn');
const runBtn = $id('runBtn');
const pngImg = $id('pngImg');
const BASE = "";

function setMsg(txt, cls = "") {
    statusBox.textContent = txt;
    statusBox.className = cls;
}

let nv1 = null;
let nv2 = null;

// 初始化 Niivue 实例
try {
    let NiivueConstructor = (typeof Niivue !== 'undefined') ? Niivue : (typeof niivue !== 'undefined' ? niivue.Niivue : null);
    
    if (NiivueConstructor) {
        const config = {
            backColor: [0, 0, 0, 1], 
            sliceType: 0,              // 0 = Axial 横断位
            isResizeCanvas: true       // 自动适应 Canvas 容器大小
        };
        
        nv1 = new NiivueConstructor(config); 
        nv1.attachTo('c1');

        nv2 = new NiivueConstructor(config);
        nv2.attachTo('c2');
        
        console.log("Niivue 0.69.0 UMD 本地图形库组件初始化完全成功！");
    } else {
        setMsg("未能正确识别 Niivue 组件，请检查本地 niivue.umd.js 是否放置正确", "error");
    }
} catch (err) {
    console.error("Niivue 初始化发生异常错误:", err);
}

// 【完美修复】上传文件时，使用 0.69.0 版本的 API 动态加载本地原图预览
fileInp.onchange = async function() {
    let file = fileInp.files[0];
    if (!file) return;
    
    setMsg("正在本地解析 3D 图像以加载预览...", "loading");
    if (!nv1) {
         setMsg("已选择文件: " + file.name + "（3D预览库未就绪）", "success");
         return;
    }

    try {
        // 适配 Niivue 0.69.0 的本地文件加载：
        // 1. 创建临时的本地 Object URL
        const blobUrl = URL.createObjectURL(file);
        
        // 2. 传入对象数组，用 url 键指向这个 blobUrl，并通过 name 指定文件名扩展名以供解析器识别
        await nv1.loadVolumes([{ url: blobUrl, name: file.name }]);
        
        setMsg("本地原图预览加载成功！可使用鼠标滚轮进行切片查看", "success");
    } catch (e) {
        setMsg("本地 3D 预览失败，您可以直接点击按钮上传分割: " + e.message, "error");
        console.error(e);
    }
};

// 检测接口状态
checkBtn.onclick = async function() {
    let k = apiKey.value.trim();
    if (!k) return setMsg("填写密钥", "error");
    setMsg("检测中...", "loading");
    try {
        let res = await fetch(BASE + "/get_server_status", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({api_key: k})
        });
        let d = await res.json();
        res.ok ? setMsg("服务正常", "success") : setMsg(d.message, "error");
    } catch(e) {
        setMsg("接口连接失败", "error");
    }
};

// 上传与 GPU 分割
runBtn.onclick = async function() {
    let key = apiKey.value.trim();
    let f = fileInp.files[0];
    let stat = ckStat.checked ? "1" : "0";
    if (!key || !f) return setMsg("密钥/文件缺失", "error");
    
    runBtn.disabled = true;
    setMsg("GPU 推理运行中，请耐心等待 (约需 10-30 秒)...", "loading");
    
    let fd = new FormData();
    fd.append("api_key", key);
    fd.append("statistics", stat);
    fd.append("data_binary", f);
    
    try {
        let res = await fetch(BASE + "/predict_image", { method: "POST", body: fd });
        if (!res.ok) {
            let err = await res.json();
            setMsg(err.message, "error");
            runBtn.disabled = false;
            return;
        }
        
        let tid = res.headers.get("X-Task-Id");
        if (!tid) {
            setMsg("无法获取任务ID，后端未正确配置跨域响应头", "error");
            runBtn.disabled = false;
            return;
        }
        
        // 触发常规 zip 下载
        let blob = await res.blob();
        let a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `seg_${tid}.zip`;
        a.click();
        
        // 1. 动态更新最下方全景静态图
        pngImg.src = `${BASE}/preview/${tid}/preview_total.png`;
        
        // 2. 推理完成后，最右侧加载分割好的标签图
        if (nv2) {
            setMsg("服务器分割成功，正在拉取 3D 分割标签...", "loading");
            const segUrl = `${BASE}/preview/${tid}/seg.nii.gz`;
            
            // 异步请求后端生成的 seg.nii.gz，并指定 roi_total 调色盘将不同器官渲染为不同颜色
            await nv2.loadVolumes([{ url: segUrl, colormap: "roi_total" }]);
            setMsg("全部数据加载完成！两边均可使用鼠标滚轮互动", "success");
        } else {
            setMsg("分割完成，压缩包已下载", "success");
        }
        
    } catch (e) {
        setMsg("请求发生异常，请检查浏览器控制台", "error");
        console.error(e);
    }
    runBtn.disabled = false;
};
</script>
</body>
</html>
"""
    return html


@app.route('/get_server_status', methods=["POST"])
def get_server_status():
    meta = request.json
    if not has_valid_credentials(meta["api_key"]):
        return {"message": "invalid access code"}, 401
    return {"status": "happily running"}, 200


@app.route('/predict_image', methods=['POST'])
def upload_data():
    meta = request.form.to_dict()
    if not has_valid_credentials(meta["api_key"]):
        return {"message": "invalid access code"}, 401

    stats = "-s" if "statistics" in meta and meta["statistics"] == "1" else ""
    img_id = str(int(time.time()))
    img_fn = f"{img_id}.nii.gz"
    src_save = UPLOAD_DIR / img_fn
    request.files['data_binary'].save(src_save)
    log(f"upload successful: {img_fn}")

    seg_dir = UPLOAD_DIR / ('seg_' + img_id)
    seg_dir.mkdir(exist_ok=True)

    # GPU推理
    cmd = f"TotalSegmentator -i {src_save} -o {seg_dir} -f -p {stats} -ns 1 -d gpu"
    subprocess.call(cmd, shell=True)

    task_pre = TMP_PREVIEW / img_id
    task_pre.mkdir(exist_ok=True)

    # 原图复制到预览目录 (前端可以动态加载此原图进行本地动态交互)
    shutil.copy(src_save, task_pre / "source.nii.gz")

    # 1. 合并所有器官为单个 seg 标签图
    try:
        import nibabel as nib
        import numpy as np
        nii_list = list(seg_dir.glob("*.nii.gz"))
        if len(nii_list):
            ref_img = nib.load(nii_list[0])
            seg_arr = np.zeros(ref_img.shape, dtype=np.uint8)
            for idx, nii_path in enumerate(nii_list, start=1):
                arr = nib.load(nii_path).get_fdata()
                seg_arr[arr > 0] = idx

            out_seg_gz = task_pre / "seg.nii.gz"
            out_seg_raw = task_pre / "seg.nii"

            # 保存合并后的图像到预览目录
            nib.save(nib.Nifti1Image(seg_arr, ref_img.affine), out_seg_gz)

            # 解压裸 nii (供不需要解压支持的旧版前端使用)
            import gzip
            with gzip.open(out_seg_gz, "rb") as fi, open(out_seg_raw, "wb") as fo:
                fo.write(fi.read())

            # 复制进打包文件夹，打入最终用户下载的 zip
            shutil.copy(out_seg_raw, seg_dir / "seg.nii")
            shutil.copy(out_seg_gz, seg_dir / "seg.nii.gz")
            log(f"{img_id}：多器官合并seg成功")
    except Exception as e:
        log(f"合并异常 {img_id}:{str(e)}")

    # 2. 寻找 TS 自动生成的 preview_total.png，复制到预览目录供前端直接 <img> 标签展示
    png_file = seg_dir / "preview_total.png"
    if png_file.exists():
        shutil.copy(png_file, task_pre / "preview_total.png")
    else:
        log(f"{img_id}未生成preview_total.png")

    # 打包 zip 供浏览器触发下载
    zip_path = str(seg_dir) + ".zip"
    shutil.make_archive(seg_dir, 'zip', seg_dir)

    resp = send_file(zip_path, mimetype="application/octet-stream")
    # 核心：设置响应头告知前端当前的任务 ID，方便前端拼接请求预览的 URL
    resp.headers["X-Task-Id"] = img_id

    # 及时清理上传的临时原图和生成的临时大文件夹
    if src_save.exists():
        os.remove(src_save)
    shutil.rmtree(seg_dir, ignore_errors=True)

    @after_this_request
    def clean_tmp(resp):
        try:
            # 及时移除生成的 zip 压缩包（因为 send_file 已经把它读入内存/流中发送了）
            if os.path.exists(zip_path):
                os.remove(zip_path)

            # 【重要】异步延迟删除前端预览目录。
            # 因为前端需要在收到响应后，再发出新的 HTTP 请求去下载 source.nii.gz 或 seg.nii.gz
            # 延迟 10 分钟（600秒）可以确保用户有足够的时间在网页上交互和查看。
            def delay_del():
                time.sleep(600)
                shutil.rmtree(TMP_PREVIEW / img_id, ignore_errors=True)

            Thread(target=delay_del).start()
        except Exception as err:
            app.logger.error(err)
        return resp

    return resp


if __name__ == '__main__':
    # 建议生产或局域网测试时保持 host='0.0.0.0'
    app.run(host='0.0.0.0', debug=False, port=5000)