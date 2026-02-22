# bunkoOCR (windows)
findtextCenterNet https://github.com/lithium0003/findtextCenterNet で公開している機械学習モデルを、アプリとして使えるようにしたbunkoOCRのWindows版です。 
このプログラムは、画像からOCR(光学文字認識)を行い、テキストに変換します。
新しめのGPUがあると、非常に高速に実行できます。 

## Compile

詳細は [HOW_TO_BUILD.md](HOW_TO_BUILD.md) を参照してください。

## Run
実行時には、DLLが必要となります。
- onnxruntime onnxruntime/build/build/Windows/Release/Release　から取ってくる
- CUDA Toolkit
- cuDNN
- TensorRT
- openvino runtime/bin/intel64/Release と runtime/3rdparty/tbb/bin から取ってくる

実行フォルダに、onnxモデルが必要です。
https://huggingface.co/lithium0003/findtextCenterNet から、onnxモデルを5つダウンロードして、配置します。

bunkoOCR.exeがGUIの実行ファイルです。内部で、OCRengine.exeを呼び出して処理します。
