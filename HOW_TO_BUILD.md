# 環境構築とビルド手順 (Windows 11 + NVIDIA GPU)

## 必要なもの一覧

| カテゴリ | ツール/SDK | バージョン |
|---------|-----------|-----------|
| IDE | Visual Studio 2022 | 最新 (v17.x) |
| バージョン管理 | Git | 最新 |
| スクリプト実行 | Python | 3.10〜3.12 |
| NVIDIA | CUDA Toolkit | **12.8** |
| NVIDIA | cuDNN | **9.8** (CUDA 12 対応版) |
| NVIDIA | TensorRT | **10.9.0.34** |
| Intel | OpenVINO GenAI | **2025.0** |

> CUDA / cuDNN / TensorRT はバージョンの組み合わせが厳密です。上記以外のバージョンを混在させると ONNX Runtime のビルドが失敗する可能性があります。

---

## Step 1: Visual Studio 2022 のインストール

1. https://visualstudio.microsoft.com/ から Community 版以上を取得
2. インストール時に以下のワークロードを選択:
   - **C++ によるデスクトップ開発** (MSVC v143 ツールセット含む)
   - **.NET デスクトップ開発** (.NET Framework 4.8 含む)
3. 「個別のコンポーネント」から **CMake ツール for C++** も追加推奨

## Step 2: Git のインストール

https://git-scm.com/download/win からインストール

## Step 3: Python のインストール

https://www.python.org/downloads/windows/ から 3.10〜3.12 をインストール

インストール時に「Add Python to PATH」にチェックを入れること

## Step 4: CUDA Toolkit 12.8 のインストール

https://developer.download.nvidia.com/compute/cuda/12.8.1/network_installers/cuda_12.8.1_windows_network.exe からダウンロードしてインストール

インストール後に環境変数 `CUDA_PATH_V12_8` が自動設定されることを確認
(例: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`)

## Step 5: cuDNN 9.8 のインストール

https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.8.0.87_cuda12-archive.zip からダウンロード
(NVIDIA Developer アカウントが必要)

zip を展開し、中身を CUDA インストール先に上書きコピー:

```
bin\     → %CUDA_PATH_V12_8%\bin\
include\ → %CUDA_PATH_V12_8%\include\
lib\     → %CUDA_PATH_V12_8%\lib\
```

## Step 6: TensorRT 10.9.0.34 のインストール

https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/zip/TensorRT-10.9.0.34.Windows.win10.cuda-12.8.zip からダウンロード
(NVIDIA Developer アカウントが必要)

zip を **`C:\TensorRT-10.9.0.34`** に展開する

> `make_onnx.bat` がこのパスをハードコードしているため、このパスに配置すること

## Step 7: OpenVINO 2025.0 のインストール

https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.0/windows/openvino_toolkit_windows_2025.0.0.17942.1f68be9f594_x86_64.zip からダウンロード

zip をリポジトリルートの **一つ上のディレクトリ** に展開する:

```
make_onnx.bat の参照先: ..\openvino_genai_windows_2025.0.0.0_x86_64\setupvars.bat

例: リポジトリが C:\src\bunkoOCR-windows\ なら
    C:\src\openvino_genai_windows_2025.0.0.0_x86_64\ に展開する
```

## Step 8: リポジトリの準備 (サブモジュール初期化)

```bash
git submodule update --init --recursive
```

`onnxruntime` サブモジュールは非常に大きいため、完了まで時間がかかります。

## Step 9: ONNX Runtime のビルド

**Developer Command Prompt for VS 2022** を開き、リポジトリルートで実行:

```batch
make_onnx.bat
```

ビルドに 30〜120 分程度かかります。
成功すると `onnxruntime\build\Windows\Release\Release\*.dll` が生成されます。

> 通常のコマンドプロンプトでは MSVC が見つからず失敗します。必ず Developer Command Prompt を使用してください。

## Step 10: bunkoOCR ソリューションのビルド

1. `bunkoOCR.sln` を Visual Studio 2022 で開く
2. 構成を **Release | x64** に変更
3. メニュー「ビルド」→「ソリューションのビルド」を実行

成功すると以下が生成されます:

```
bunkoOCR\bin\Release\bunkoOCR.exe
OCRengine\x64\Release\OCRengine.exe
textline_detect\x64\Release\textline_detect.exe
detectGPU\x64\Release\detectGPU.exe
```

---

## 実行時の準備

`bunkoOCR.exe` と同じフォルダに以下を配置してください。

### ONNX Runtime DLL

`onnxruntime\build\Windows\Release\Release\` 内の `*.dll` をコピー

### C++ 実行ファイル

```
OCRengine.exe
textline_detect.exe
detectGPU.exe
```

### GPU 関連 DLL (NVIDIA GPU 使用時)

```
CUDA Toolkit の bin\ から cudart64_*.dll, cublas64_*.dll, cublasLt64_*.dll 等
TensorRT の bin\ から nvinfer.dll, nvonnxparser.dll 等
```

### OpenVINO DLL (Intel GPU/CPU 使用時)

```
openvino の runtime\bin\intel64\Release\ 内の *.dll
openvino の runtime\3rdparty\tbb\bin\ 内の *.dll
```

### ONNX モデルファイル (5つ)

https://huggingface.co/lithium0003/findtextCenterNet からダウンロードして配置:

```
TextDetector.onnx
CodeDecoder.onnx
Encoder.onnx
Decoder.onnx
ruby_detect.onnx
```
