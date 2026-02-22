# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

bunkoOCR-windows is a Windows OCR application for Japanese text recognition, using machine learning models from [findtextCenterNet](https://huggingface.co/lithium0003/findtextCenterNet). It consists of a C# WinForms GUI that orchestrates C++ console executables via pipes.

## Build

Open `bunkoOCR.sln` in Visual Studio 2022. The solution contains 4 projects:

| Project | Type | Language |
|---------|------|----------|
| `bunkoOCR` | WinExe (.NET 4.8) | C# |
| `OCRengine` | Console app | C++ (C++20) |
| `textline_detect` | Console app | C++ (C++20) |
| `detectGPU` | Console app | C++ (C++20) |

**Prerequisites before building C++ projects:**
1. Build ONNX Runtime from the submodule using `make_onnx.bat` (requires CUDA 12.8, cuDNN 9.8, TensorRT 10.9, OpenVINO 2025.0 installed)
2. ONNX Runtime DLLs must be present at `onnxruntime/build/Windows/Release/Release/`

## Architecture

### Component Relationships

```
bunkoOCR.exe (C# WinForms GUI)
  └─ spawns OCRengine.exe via stdin/stdout pipes
       ├─ TextDetector.onnx    → detects character locations (768×768 → 192×192 heatmap)
       ├─ CodeDecoder.onnx     → decodes character codes from glyph features
       ├─ Encoder.onnx         → transformer encoder (100-dim features → 512-dim)
       ├─ Decoder.onnx         → transformer decoder → modulo outputs for char prediction
       └─ spawns textline_detect.exe → groups characters into lines/blocks, detects ruby
  └─ spawns detectGPU.exe → queries DXGI to find best DirectML GPU index
```

### GUI Layer ([bunkoOCR/](bunkoOCR/))

- [Form1.cs](bunkoOCR/Form1.cs) — Main form: drag-drop queue, spawns `OCRengine.exe` subprocess, sends image filenames via stdin, receives JSON results via stdout
- [Form2.cs](bunkoOCR/Form2.cs) — Secondary dialog
- [Form3.cs](bunkoOCR/Form3.cs) — Settings panel for all tunable detection parameters
- [ConfigReader.cs](bunkoOCR/ConfigReader.cs) — Loads/saves `param.config` and `ruby.config`

### OCR Engine ([OCRengine/](OCRengine/))

- [OCRengine.cpp](OCRengine/OCRengine.cpp) — Entry point; reads filenames from stdin, outputs JSON to stdout and result files
- [TextDetector.cpp](OCRengine/TextDetector.cpp) / [.h](OCRengine/TextDetector.h) — Runs CenterNet ONNX model to detect text locations and glyph features
- [CodeDecoder.cpp](OCRengine/CodeDecoder.cpp) / [.h](OCRengine/CodeDecoder.h) — Decodes character codes from detected features
- [Transformer.cpp](OCRengine/Transformer.cpp) / [.h](OCRengine/Transformer.h) — Encoder-decoder transformer for sequence recognition
- [line_detect.cpp](OCRengine/line_detect.cpp) / [.h](OCRengine/line_detect.h) — Groups detected characters into lines and text blocks

### ONNX Runtime Execution Providers (priority order)

1. TensorRT (NVIDIA)
2. CUDA (NVIDIA)
3. DirectML (Windows — AMD/Intel/NVIDIA via D3D12)
4. OpenVINO (Intel)
5. CPU fallback

### IPC Protocol

Form1 ↔ OCRengine communicate via stdin/stdout pipes:
- **Input to OCRengine:** config key:value pairs, then image file paths
- **Output from OCRengine:** JSON with detected text, coordinates, ruby annotations, formatting metadata; status lines like `done: filename` or `error: filename`

## Configuration

`param.config` (key=value format):
- GPU: `use_TensorRT`, `use_CUDA`, `use_DirectML`, `use_OpenVINO`, `DML_GPU_id` (-1 = auto)
- Detection thresholds: `detect_cut_off`, `blank_cutoff`, `ruby_cutoff`, `rubybase_cutoff`
- Line detection: `line_valueth`, `sep_valueth`, `sep_valueth2`, `allowwidth_next`
- Processing: `resize`, `sleep_wait`, `autostart`

`ruby.config` (key=value format):
- Output format: `raw_output`, `output_text`, `output_ruby`
- Ruby delimiters: `before_ruby`, `separator_ruby`, `after_ruby` (e.g., Aozora format: `｜《》`)

## Runtime Requirements

Place in the same directory as the executables:
- ONNX Runtime DLLs
- CUDA / cuDNN / TensorRT DLLs (if using NVIDIA GPU)
- OpenVINO runtime DLLs (if using Intel)
- 5 ONNX model files from https://huggingface.co/lithium0003/findtextCenterNet
