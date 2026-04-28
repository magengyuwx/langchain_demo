import argparse
import json
import os
import shutil
import threading
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from txt2json import ALLOWED_IMAGE_PARTS
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_ROOT = r"D:\git\LV-Dataset\products"

@dataclass
class ViewerConfig:
    dataset_root: Path
    host: str
    port: int


def _resolve_product_dir(root: Path, product_name: str) -> Path:
    candidate_name = product_name.strip() or "."
    candidate = (root / candidate_name).resolve()
    root_resolved = root.resolve()
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise ValueError("非法目录路径")
    if not candidate.is_dir():
        raise FileNotFoundError(f"商品目录不存在: {candidate}")
    return candidate


def _list_products(root: Path) -> list[str]:
    return [child.name for child in sorted(root.iterdir()) if child.is_dir()]


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    for encoding in ("utf-8", "gb18030", "latin1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _read_annotation(path: Path) -> dict:
    if not path.exists():
        return {"images": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_raw_error": "annotation.json 不是合法 JSON", "images": []}


def _list_images(product_dir: Path) -> list[str]:
    images = [p.name for p in product_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(images)


def _merge_images(annotation: dict, files: list[str]) -> list[dict]:
    existing = {}
    images = annotation.get("images", [])
    if isinstance(images, list):
        for item in images:
            if isinstance(item, dict):
                name = str(item.get("filename", "")).strip()
                if name:
                    existing[name] = item

    result = []
    for filename in files:
        row = existing.get(filename, {})
        result.append(
            {
                "filename": filename,
                "part": str(row.get("part", "")).strip(),
                "description": str(row.get("description", "")).strip(),
            }
        )
    return result


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dataset Annotation Viewer</title>
  <style>
    :root {
      --bg: #f6f7f8;
      --panel: #ffffff;
      --line: #e5e7eb;
      --text: #111827;
      --muted: #6b7280;
      --brand: #165dff;
      --ok: #098551;
      --warn: #b45309;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft Yahei", sans-serif;
      color: var(--text);
      background: linear-gradient(120deg, #eef2ff 0%, #f8fafc 45%, #fef7ed 100%);
    }
    .layout {
      display: grid;
      grid-template-columns: 300px 1fr;
      min-height: 100vh;
      gap: 12px;
      padding: 12px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
    }
    .sidebar { display: flex; flex-direction: column; }
    .header {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      font-weight: 700;
    }
    .body { padding: 12px; }
    .muted { color: var(--muted); font-size: 12px; }
    .product-list {
      list-style: none;
      margin: 0;
      padding: 0;
      max-height: calc(100vh - 160px);
      overflow: auto;
    }
    .product-list li {
      padding: 8px 10px;
      border-radius: 8px;
      margin-bottom: 6px;
      cursor: pointer;
      border: 1px solid transparent;
    }
    .product-list li:hover { background: #f3f4f6; }
    .product-list li.active {
      background: #eff6ff;
      border-color: #bfdbfe;
      color: #1d4ed8;
      font-weight: 600;
    }
    .content {
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      gap: 12px;
      padding: 12px;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    button {
      border: 1px solid #cbd5e1;
      background: #fff;
      padding: 8px 12px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
    }
    button.primary {
      background: var(--brand);
      border-color: var(--brand);
      color: #fff;
    }
    .status {
      margin-left: auto;
      font-size: 12px;
      color: var(--muted);
    }
    .grid {
      display: grid;
      gap: 12px;
      grid-template-columns: 1fr 1fr;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
      background: #fff;
    }
    textarea, select {
      width: 100%;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 8px;
      font: inherit;
      background: #fff;
    }
    textarea { min-height: 120px; resize: vertical; }
    #jsonEditor { min-height: 280px; font-family: Consolas, monospace; }
    .image-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 12px;
      max-height: none;
      overflow: visible;
      padding-right: 0;
    }
    .img-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      background: #fff;
    }
    .img-card img {
      width: 100%;
      height: 180px;
      object-fit: cover;
      border-radius: 8px;
      background: #f3f4f6;
    }
    .img-card .name {
      margin: 8px 0 6px 0;
      font-weight: 600;
      font-size: 13px;
      word-break: break-all;
    }
    .img-card textarea { min-height: 80px; font-size: 13px; }
    @media (max-width: 1024px) {
      .layout { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
      .image-grid { max-height: none; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <section class="panel sidebar">
      <div class="header">商品目录</div>
      <div class="body">
        <div class="muted">根目录: <span id="rootPath"></span></div>
        <button id="refreshBtn" style="margin:8px 0 10px 0;">刷新目录</button>
        <ul id="productList" class="product-list"></ul>
      </div>
    </section>
    <section class="panel content">
      <div id="statusText" class="muted">未加载</div>
      <div class="card">
        <div style="font-weight:700; margin-bottom:8px;">图片标注编辑</div>
        <div id="imageGrid" class="image-grid"></div>
      </div>
      <div class="grid">
        <div class="card">
          <div style="font-weight:700; margin-bottom:6px;">info.txt</div>
          <textarea id="infoText" readonly></textarea>
        </div>
        <div class="card">
          <div style="font-weight:700; margin-bottom:6px;">annotation.json（只读）</div>
          <textarea id="jsonEditor" readonly></textarea>
        </div>
      </div>
      <div class="muted">提示: 修改图片部位或描述后会自动同步并保存 annotation.json。</div>
    </section>
  </div>
  <script>
    const allowedParts = __ALLOWED_PARTS__;
    let currentProduct = "";
    let autoSaveTimer = null;

    const rootPathEl = document.getElementById("rootPath");
    const productListEl = document.getElementById("productList");
    const infoTextEl = document.getElementById("infoText");
    const jsonEditorEl = document.getElementById("jsonEditor");
    const imageGridEl = document.getElementById("imageGrid");
    const statusTextEl = document.getElementById("statusText");

    function setStatus(text, color = "") {
      statusTextEl.textContent = text;
      statusTextEl.style.color = color || "";
    }

    async function loadProducts() {
      const resp = await fetch("/api/products");
      const data = await resp.json();
      rootPathEl.textContent = data.root;
      productListEl.innerHTML = "";
      data.products.forEach((name) => {
        const li = document.createElement("li");
        li.textContent = name;
        li.onclick = () => selectProduct(name, li);
        productListEl.appendChild(li);
      });
      if (data.products.length > 0) {
        const first = productListEl.querySelector("li");
        selectProduct(data.products[0], first);
      } else {
        setStatus("目录下没有可用商品子目录", "#b45309");
      }
    }

    function setActiveLi(target) {
      productListEl.querySelectorAll("li").forEach((li) => li.classList.remove("active"));
      if (target) target.classList.add("active");
    }

    async function selectProduct(name, liNode) {
      try {
        setActiveLi(liNode);
        setStatus(`加载中: ${name}...`);
        const resp = await fetch(`/api/product?name=${encodeURIComponent(name)}`);
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "加载失败");
        currentProduct = name;
        infoTextEl.value = data.info_text || "";
        jsonEditorEl.value = JSON.stringify(data.annotation, null, 2);
        renderImageCards(data.images);
        setStatus(`已加载: ${name}`, "#098551");
      } catch (err) {
        setStatus(err.message || "加载失败", "#b91c1c");
      }
    }

    function renderImageCards(images) {
      imageGridEl.innerHTML = "";
      images.forEach((item) => {
        const card = document.createElement("div");
        card.className = "img-card";
        card.innerHTML = `
          <img src="/api/image?name=${encodeURIComponent(currentProduct)}&file=${encodeURIComponent(item.filename)}" alt="${item.filename}" />
          <div class="name">${item.filename}</div>
          <select class="part"></select>
          <textarea class="desc" placeholder="图片描述">${item.description || ""}</textarea>
        `;
        const select = card.querySelector(".part");
        const emptyOpt = document.createElement("option");
        emptyOpt.value = "";
        emptyOpt.textContent = "(未标注)";
        select.appendChild(emptyOpt);
        allowedParts.forEach((part) => {
          const opt = document.createElement("option");
          opt.value = part;
          opt.textContent = part;
          if (item.part === part) opt.selected = true;
          select.appendChild(opt);
        });
        card.dataset.filename = item.filename;
        imageGridEl.appendChild(card);
      });
    }

    function safeParseJson() {
      try {
        return JSON.parse(jsonEditorEl.value);
      } catch {
        throw new Error("JSON 格式错误，请先修正 annotation.json 内容");
      }
    }

    function syncCardsToJson() {
      const obj = safeParseJson();
      const rows = Array.from(imageGridEl.querySelectorAll(".img-card")).map((card) => ({
        filename: card.dataset.filename || "",
        part: card.querySelector(".part").value || "",
        description: card.querySelector(".desc").value || "",
      }));
      obj.images = rows;
      jsonEditorEl.value = JSON.stringify(obj, null, 2);
      return obj;
    }

    function queueAutoSave(delayMs = 500) {
      if (!currentProduct) return;
      try {
        syncCardsToJson();
      } catch (err) {
        setStatus(err.message || "同步失败", "#b91c1c");
        return;
      }
      setStatus("已更新，自动保存中...", "#b45309");
      if (autoSaveTimer) {
        clearTimeout(autoSaveTimer);
      }
      autoSaveTimer = setTimeout(() => {
        autoSaveTimer = null;
        saveAnnotation();
      }, delayMs);
    }

    async function saveAnnotation() {
      try {
        if (!currentProduct) throw new Error("请先选择一个商品目录");
        const obj = safeParseJson();
        setStatus("保存中...");
        const resp = await fetch(`/api/save?name=${encodeURIComponent(currentProduct)}`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(obj),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "保存失败");
        setStatus(`保存成功: ${data.path}`, "#098551");
      } catch (err) {
        setStatus(err.message || "保存失败", "#b91c1c");
      }
    }

    document.getElementById("refreshBtn").onclick = loadProducts;
    imageGridEl.addEventListener("change", (event) => {
      if (event.target.closest(".img-card")) {
        queueAutoSave(0);
      }
    });
    imageGridEl.addEventListener("input", (event) => {
      if (event.target.classList && event.target.classList.contains("desc")) {
        queueAutoSave(700);
      }
    });

    loadProducts().catch((e) => setStatus(e.message || "加载失败", "#b91c1c"));
  </script>
</body>
</html>"""


def _json_response(handler: BaseHTTPRequestHandler, data: dict, status: int = 200) -> None:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class ViewerHandler(BaseHTTPRequestHandler):
    config: ViewerConfig
    lock = threading.Lock()

    def _parse_query(self) -> dict[str, str]:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        return {k: v[0] for k, v in qs.items() if v}

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "文件不存在")
            return
        data = file_path.read_bytes()
        ext = file_path.suffix.lower()
        content_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(ext, "application/octet-stream")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        root = self.config.dataset_root

        try:
            if route == "/":
                html = HTML_PAGE.replace("__ALLOWED_PARTS__", json.dumps(ALLOWED_IMAGE_PARTS, ensure_ascii=False))
                self._send_html(html)
                return

            if route == "/api/products":
                _json_response(self, {"root": str(root), "products": _list_products(root)})
                return

            if route == "/api/product":
                query = self._parse_query()
                name = query.get("name", "").strip()
                if not name:
                    _json_response(self, {"error": "缺少参数 name"}, status=400)
                    return
                product_dir = _resolve_product_dir(root, name)
                info_text = _read_text(product_dir / "info.txt")
                annotation = _read_annotation(product_dir / "annotation.json")
                files = _list_images(product_dir)
                merged_images = _merge_images(annotation, files)
                _json_response(
                    self,
                    {"name": name, "info_text": info_text, "annotation": annotation, "images": merged_images},
                )
                return

            if route == "/api/image":
                query = self._parse_query()
                name = query.get("name", "").strip()
                file_name = query.get("file", "").strip()
                if not name or not file_name:
                    self.send_error(HTTPStatus.BAD_REQUEST, "缺少参数")
                    return
                product_dir = _resolve_product_dir(root, name)
                image_path = (product_dir / file_name).resolve()
                if product_dir.resolve() not in image_path.parents:
                    self.send_error(HTTPStatus.BAD_REQUEST, "非法文件路径")
                    return
                self._send_file(image_path)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
        except Exception as exc:
            _json_response(self, {"error": str(exc)}, status=500)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        if route != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        query = self._parse_query()
        name = query.get("name", "").strip()
        if not name:
            _json_response(self, {"error": "缺少参数 name"}, status=400)
            return

        try:
            content_len = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_len = 0
        raw = self.rfile.read(content_len) if content_len > 0 else b""
        if not raw:
            _json_response(self, {"error": "请求体为空"}, status=400)
            return

        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            _json_response(self, {"error": "请求体不是合法 JSON"}, status=400)
            return

        try:
            product_dir = _resolve_product_dir(self.config.dataset_root, name)
            annotation_path = product_dir / "annotation.json"
            backup_path = annotation_path.with_suffix(f".json.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            with self.lock:
                if annotation_path.exists():
                    shutil.copy2(annotation_path, backup_path)
                annotation_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            _json_response(
                self,
                {
                    "message": "保存成功",
                    "path": str(annotation_path),
                    "backup_path": str(backup_path) if backup_path.exists() else "",
                },
            )
        except Exception as exc:
            _json_response(self, {"error": str(exc)}, status=500)

    def log_message(self, format: str, *args) -> None:
        return


def start_server(config: ViewerConfig) -> None:
    handler_cls = type("BoundViewerHandler", (ViewerHandler,), {"config": config})
    with ThreadingHTTPServer((config.host, config.port), handler_cls) as server:
        print(f"Web Viewer 已启动: http://{config.host}:{config.port}")
        print(f"数据目录: {config.dataset_root}")
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="浏览并编辑商品 annotation.json 的本地网页工具")
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="数据集根目录（每个商品一个子目录）",
    )
    parser.add_argument("--host", default="127.0.0.1", help="服务监听地址")
    parser.add_argument("--port", type=int, default=8899, help="服务端口")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {root}")

    config = ViewerConfig(dataset_root=root, host=args.host, port=args.port)
    start_server(config)


if __name__ == "__main__":
    main()
