#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
bunkoOCR CLI - OCRengine.exe を使って画像から日本語テキストを認識するコマンドラインツール。

bunkoOCR.exe (C# WinForms GUI) と同等の処理を GUI なしで実行する。
param.config / ruby.config / path.config は bunkoOCR.exe で作成済みのものを利用する。

使用例:
    uv run bunko_ocr.py image1.jpg image2.jpg
    uv run bunko_ocr.py --engine-dir C:\\path\\to\\ocr image.jpg
"""

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path


# ---------------------------------------------------------------------------
# 設定ファイル読み込み
# ---------------------------------------------------------------------------

def _parse_config_file(path: Path) -> dict[str, str]:
    """key:value 形式の設定ファイルを読み込む。最初の ':' で分割する。"""
    result: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            pos = line.find(":")
            if pos < 0:
                continue
            key = line[:pos]
            val = line[pos + 1:]
            result[key] = val
    except OSError:
        pass
    return result


def load_param_config(config_dir: Path) -> dict[str, float]:
    """param.config を読み込み、数値パラメータの辞書を返す。"""
    defaults: dict[str, float] = {
        "detect_cut_off": 0.35,
        "blank_cutoff": 20.0,
        "ruby_cutoff": 0.25,
        "rubybase_cutoff": 0.75,
        "space_cutoff": 0.5,
        "emphasis_cutoff": 0.5,
        "line_valueth": 0.25,
        "sep_valueth": 0.15,
        "sep_valueth2": 0.2,
        "allowwidth_next": 1.0,
        "resize": 1.0,
        "sleep_wait": 0.0,
        "use_TensorRT": 1.0,
        "use_CUDA": 1.0,
        "use_DirectML": 1.0,
        "DML_GPU_id": -1.0,
        "use_OpenVINO": 1.0,
        "autostart": 1.0,
    }
    raw = _parse_config_file(config_dir / "param.config")
    for key, val in raw.items():
        try:
            defaults[key] = float(val)
        except ValueError:
            pass
    return defaults


def load_ruby_config(config_dir: Path) -> dict[str, str]:
    """ruby.config を読み込み、ルビ出力設定の辞書を返す。"""
    defaults: dict[str, str] = {
        "raw_output": "0",
        "output_text": "1",
        "output_ruby": "1",
        "before_ruby": "\uff5c",    # ｜
        "separator_ruby": "\u300a", # 《
        "after_ruby": "\u300b",     # 》
    }
    raw = _parse_config_file(config_dir / "ruby.config")
    defaults.update(raw)
    return defaults


def load_path_config(config_dir: Path) -> dict[str, str]:
    """path.config を読み込み、出力パス設定の辞書を返す。"""
    defaults: dict[str, str] = {
        "output_dir": "",
        "override": "0",
    }
    raw = _parse_config_file(config_dir / "path.config")
    defaults.update(raw)
    return defaults


# ---------------------------------------------------------------------------
# OCRengine.exe 引数の構築
# ---------------------------------------------------------------------------

# エンジンに送信しない GUI 専用キー（ConfigReader.FilterForOCRengine の再現）
_ENGINE_EXCLUDED_KEYS = {
    "use_TensorRT", "use_CUDA", "use_DirectML", "use_OpenVINO", "DML_GPU_id", "autostart"
}


def filter_for_engine(params: dict[str, float]) -> dict[str, float]:
    """エンジンに送信しないキーを除外する。"""
    return {k: v for k, v in params.items() if k not in _ENGINE_EXCLUDED_KEYS}


def run_detect_gpu(engine_dir: Path) -> int:
    """detectGPU.exe を実行し、終了コード（GPU インデックス）を返す。"""
    exe = engine_dir / "detectGPU.exe"
    if not exe.exists():
        return -1
    try:
        result = subprocess.run(
            [str(exe)],
            capture_output=True,
            cwd=str(engine_dir),
        )
        return result.returncode
    except OSError:
        return -1


def build_engine_args(params: dict[str, float], engine_dir: Path) -> list[str]:
    """OCRengine.exe に渡すコマンドライン引数リストを構築する。"""
    args: list[str] = []
    if params.get("use_TensorRT", 0) > 0:
        args.append("TensorRT")
    if params.get("use_CUDA", 0) > 0:
        args.append("CUDA")
    if params.get("use_OpenVINO", 0) > 0:
        args.append("OpenVINO")
    if params.get("use_DirectML", 0) > 0:
        gpu_id = int(params.get("DML_GPU_id", -1))
        if gpu_id < 0 or gpu_id > 254:
            # DML_GPU_id=-1 の場合は detectGPU.exe で自動検出
            gpu_id = run_detect_gpu(engine_dir)
        if 0 <= gpu_id <= 254:
            args.append(str(gpu_id))
    return args


# ---------------------------------------------------------------------------
# 出力パスの解決
# ---------------------------------------------------------------------------

def resolve_output_path(
    input_path: Path,
    output_dir: str,
    override: bool,
) -> Path:
    """
    出力 JSON ファイルのパスを決定する（Form1.cs SendToEngine の再現）。

    戻り値が Path("") の場合は空文字列をエンジンに送信し、
    エンジンが input_path + ".json" をデフォルトとして使用する。
    """
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        candidate = out_dir / (input_path.name + ".json")
        if candidate.exists() and not override:
            n = 1
            while candidate.exists():
                candidate = out_dir / (input_path.name + f".{n}.json")
                n += 1
        return candidate
    elif not override:
        candidate = Path(str(input_path) + ".json")
        if candidate.exists():
            n = 1
            while candidate.exists():
                candidate = Path(str(input_path) + f".{n}.json")
                n += 1
        return candidate
    else:
        # override=True, output_dir="" → 空文字列をエンジンに送信
        return Path("")


# ---------------------------------------------------------------------------
# ポストプロセス（JSON のルビ変換・txt 出力）
# ---------------------------------------------------------------------------

_RUBY_PATTERN = re.compile(r"\uFFF9(.*?)\uFFFA(.*?)\uFFFB")


def _apply_ruby(text: str, output_ruby: bool, before: str, sep: str, after: str) -> str:
    """ルビ制御文字（U+FFF9〜U+FFFB）を設定に応じた文字列に置換する。"""
    if output_ruby:
        return _RUBY_PATTERN.sub(
            lambda m: before + m.group(1) + sep + m.group(2) + after,
            text,
        )
    else:
        return _RUBY_PATTERN.sub(
            lambda m: before + m.group(1) + after,
            text,
        )


def postprocess(input_file: str, json_path: Path, ruby_cfg: dict[str, str]) -> None:
    """
    OCRengine が出力した JSON を読み込み、ルビ変換・再書き込み・txt 出力を行う
    （Form1.cs postprocess の再現）。
    """
    raw_output = ruby_cfg.get("raw_output") == "1"
    output_text = ruby_cfg.get("output_text") == "1"
    output_ruby = ruby_cfg.get("output_ruby") == "1"
    before_ruby = ruby_cfg.get("before_ruby", "\uff5c")
    separator_ruby = ruby_cfg.get("separator_ruby", "\u300a")
    after_ruby = ruby_cfg.get("after_ruby", "\u300b")

    if not json_path.exists():
        print(f"WARNING: JSON ファイルが見つかりません: {json_path}", file=sys.stderr)
        return

    with open(json_path, encoding="utf-8") as f:
        result = json.load(f)

    if not raw_output:
        def proc(s: str) -> str:
            return _apply_ruby(s, output_ruby, before_ruby, separator_ruby, after_ruby)

        if "text" in result:
            result["text"] = proc(result["text"])
        for block in result.get("block", []):
            if "text" in block:
                block["text"] = proc(block["text"])
        for line in result.get("line", []):
            if "text" in line:
                line["text"] = proc(line["text"])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if output_text:
        txt_path = json_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result.get("text", ""))
        print(f"  → {txt_path}", file=sys.stderr)

    print(f"  → {json_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 環境チェック（--doctor）
# ---------------------------------------------------------------------------

def run_doctor(engine_dir: Path, config_dir: Path) -> int:
    """
    動作に必要なファイルの存在を確認し、結果を stdout に表示する。
    必須ファイルがすべて揃っていれば 0、不足があれば 1 を返す。
    """
    # (filename, required, note)
    engine_files: list[tuple[str, bool, str]] = [
        ("OCRengine.exe",          True,  ""),
        ("textline_detect.exe",    True,  ""),
        ("detectGPU.exe",          False, "任意: DirectML の GPU 自動検出に使用"),
        ("TextDetector.onnx",      False, "TextDetector.quant.onnx との either/or"),
        ("TextDetector.quant.onnx",False, "任意: TextDetector.onnx の量子化版（代替）"),
        ("CodeDecoder.onnx",       True,  ""),
        ("TransformerEncoder.onnx",True,  ""),
        ("TransformerDecoder.onnx",True,  ""),
    ]
    config_files: list[tuple[str, bool, str]] = [
        ("param.config", False, "任意: なければデフォルト値を使用"),
        ("ruby.config",  False, "任意: なければデフォルト値を使用"),
        ("path.config",  False, "任意: なければデフォルト値を使用"),
    ]

    print("[doctor] 環境チェック")
    missing_required = 0

    # --- engine-dir ---
    print(f"\nengine-dir: {engine_dir}")
    td_onnx   = (engine_dir / "TextDetector.onnx").exists()
    td_quant  = (engine_dir / "TextDetector.quant.onnx").exists()

    for filename, required, note in engine_files:
        exists = (engine_dir / filename).exists()

        # TextDetector 系は either/or で必須判定
        if filename == "TextDetector.onnx":
            is_missing_required = (not td_onnx and not td_quant)
        elif filename == "TextDetector.quant.onnx":
            is_missing_required = False  # こちら側では必須扱いしない
        else:
            is_missing_required = required and not exists

        if exists:
            tag = "[OK]     "
        elif required or (filename == "TextDetector.onnx" and not td_quant):
            tag = "[MISSING]"
        else:
            tag = "[--]     "

        suffix = f"  ({note})" if note else ""
        print(f"  {tag} {filename}{suffix}")

        if is_missing_required:
            missing_required += 1

    # TextDetector: 両方なければ1件のエラーとして集計（上のループ内で計上済み）

    # --- config-dir ---
    same_dir = engine_dir.resolve() == config_dir.resolve()
    label = f"config-dir: {config_dir}" + ("  (engine-dir と同じ)" if same_dir else "")
    print(f"\n{label}")
    for filename, required, note in config_files:
        exists = (config_dir / filename).exists()
        tag = "[OK]     " if exists else "[--]     "
        suffix = f"  ({note})" if note else ""
        print(f"  {tag} {filename}{suffix}")
        if required and not exists:
            missing_required += 1

    # --- 結果サマリ ---
    print()
    if missing_required == 0:
        print("結果: OK (必須ファイルはすべて揃っています)")
        return 0
    else:
        print(f"結果: NG (必須ファイルが {missing_required} 件不足しています)")
        return 1


# ---------------------------------------------------------------------------
# stdout 読み込みスレッド
# ---------------------------------------------------------------------------

def _stdout_reader(proc: subprocess.Popen, out_queue: queue.Queue, verbose: bool) -> None:
    """OCRengine の stdout を行単位で読み込み、キューに積む。"""
    for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if verbose:
            print(f"[engine] {line}", file=sys.stderr)
        out_queue.put(line)
    out_queue.put(None)  # 終端シグナル


# ---------------------------------------------------------------------------
# メインパイプライン
# ---------------------------------------------------------------------------

def run_ocr(
    image_files: list[str],
    engine_dir: Path,
    config_dir: Path,
    output_dir: str,
    override: bool,
    verbose: bool,
) -> int:
    """OCR パイプラインを実行する。成功なら 0、失敗があれば 1 を返す。"""

    # 設定読み込み
    params = load_param_config(config_dir)
    ruby_cfg = load_ruby_config(config_dir)

    # エンジン引数構築
    engine_args = build_engine_args(params, engine_dir)
    engine_params = filter_for_engine(params)

    # stdin に送信する設定パラメータ文字列
    config_bytes = "".join(
        f"{k}:{v}\r\n" for k, v in engine_params.items()
    ).encode("utf-8")

    # 入力画像のチェックと出力パス解決
    file_map: dict[str, Path] = {}  # 送信する input_str → output_path
    for img in image_files:
        p = Path(img)
        if not p.exists():
            print(f"WARNING: ファイルが見つかりません、スキップします: {img}", file=sys.stderr)
            continue
        # 絶対パスで送信する（エンジンの cwd に依存しないよう）
        abs_input = str(p.resolve())
        out_path = resolve_output_path(p.resolve(), output_dir, override)
        file_map[abs_input] = out_path

    if not file_map:
        print("処理対象の画像ファイルがありません。", file=sys.stderr)
        return 1

    # OCRengine.exe の存在確認
    engine_exe = engine_dir / "OCRengine.exe"
    if not engine_exe.exists():
        print(f"ERROR: OCRengine.exe が見つかりません: {engine_exe}", file=sys.stderr)
        return 1

    cmd = [str(engine_exe)] + engine_args
    if verbose:
        print(f"[engine] 起動: {' '.join(cmd)}", file=sys.stderr)
        print(f"[engine] cwd: {engine_dir}", file=sys.stderr)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # エンジンの stderr はそのまま表示
        cwd=str(engine_dir),
    )

    out_queue: queue.Queue = queue.Queue()
    reader_thread = threading.Thread(
        target=_stdout_reader,
        args=(proc, out_queue, verbose),
        daemon=True,
    )
    reader_thread.start()

    # 設定パラメータを即座に送信（パイプバッファに蓄積、ready 後にエンジンが消費）
    proc.stdin.write(config_bytes)
    proc.stdin.flush()

    # "ready" を待機
    print("OCR エンジンの初期化を待っています...", file=sys.stderr)
    while True:
        line = out_queue.get()
        if line is None:
            print("ERROR: エンジンが ready になる前に終了しました。", file=sys.stderr)
            proc.wait()
            return 1
        if line == "ready":
            break

    print(f"エンジン準備完了。{len(file_map)} 件の画像を処理します...", file=sys.stderr)

    # 画像ペアを送信
    for input_str, out_path in file_map.items():
        out_str = str(out_path) if str(out_path) != "." else ""
        # output_path が Path("") の場合は空文字列を送信
        if out_path == Path(""):
            out_str = ""
        payload = f"{input_str}\r\n{out_str}\r\n".encode("utf-8")
        proc.stdin.write(payload)
        proc.stdin.flush()

    proc.stdin.close()

    # done: / error: を全件受信するまで待機
    pending = set(file_map.keys())
    done_files: list[tuple[str, Path]] = []
    error_count = 0
    total = len(pending)
    processed = 0

    while pending:
        line = out_queue.get()
        if line is None:
            # エンジンが終了した（未受信の画像がある場合はエラー）
            if pending:
                print(
                    f"ERROR: エンジンが終了しましたが {len(pending)} 件の結果を受信できませんでした。",
                    file=sys.stderr,
                )
            break

        if line.startswith("done: "):
            done_filename = line[len("done: "):]
            pending.discard(done_filename)
            processed += 1
            print(f"[{processed}/{total}] 完了: {done_filename}", file=sys.stderr)

            # 対応する出力パスを解決
            out_path = file_map.get(done_filename)
            if out_path is None:
                # フォールバック
                out_path = Path(done_filename + ".json")
            elif out_path == Path(""):
                out_path = Path(done_filename + ".json")
            done_files.append((done_filename, out_path))

        elif line.startswith("error: "):
            err_filename = line[len("error: "):]
            pending.discard(err_filename)
            error_count += 1
            processed += 1
            print(f"[{processed}/{total}] ERROR: {err_filename}", file=sys.stderr)

    proc.wait()
    reader_thread.join(timeout=5)

    # ポストプロセス
    if done_files:
        print("ポストプロセス中...", file=sys.stderr)
        for input_file, json_path in done_files:
            postprocess(input_file, json_path, ruby_cfg)

    print(
        f"完了: {len(done_files)} 件成功、{error_count} 件失敗",
        file=sys.stderr,
    )
    return 0 if error_count == 0 else 1


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "bunkoOCR CLI - OCRengine.exe を使って画像から日本語テキストを認識する\n"
            "\n"
            "param.config / ruby.config / path.config は bunkoOCR.exe で作成済みの\n"
            "ものを利用します（engine-dir に配置）。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "images",
        nargs="*",
        metavar="IMAGE",
        help="処理する画像ファイル（--doctor 使用時は不要）",
    )
    parser.add_argument(
        "--engine-dir",
        default=None,
        metavar="PATH",
        help="OCRengine.exe のあるディレクトリ（デフォルト: スクリプトと同じディレクトリ）",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        metavar="PATH",
        help="*.config ファイルのあるディレクトリ（デフォルト: engine-dir と同じ）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="出力ディレクトリ（path.config の output_dir より優先）",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="既存の出力ファイルを上書きする（path.config の override より優先）",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="OCRengine の出力を stderr に表示する",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="動作に必要なファイルの存在を確認して結果を表示する",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    engine_dir = Path(args.engine_dir) if args.engine_dir else script_dir
    config_dir = Path(args.config_dir) if args.config_dir else engine_dir

    if args.doctor:
        return run_doctor(engine_dir, config_dir)

    if not args.images:
        parser.error("処理する IMAGE を1つ以上指定してください（または --doctor で環境チェック）")

    # path.config から output_dir / override を読み込み、CLI 引数で上書き
    path_cfg = load_path_config(config_dir)
    output_dir = args.output_dir if args.output_dir is not None else path_cfg["output_dir"]
    override = args.override or (path_cfg["override"] == "1")

    return run_ocr(
        image_files=args.images,
        engine_dir=engine_dir,
        config_dir=config_dir,
        output_dir=output_dir,
        override=override,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
