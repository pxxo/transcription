# Whisper + SudachiPy サーバープロジェクト（開発者・実機サーバー向け）

このプロジェクトは、OpenAI Whisper を用いた日本語音声認識APIサーバーです。音声ファイルをアップロードすると、Whisperで文字起こしし、SudachiPyで日本語の句読点を自動付与します。React(Next.js)フロントエンドと連携し、進捗・順番待ちもリアルタイム表示されます。

## システム構成

- Ubuntu 20.04 以上（推奨）
- Python 3.8 以上
- CUDA対応GPU（推奨、なければCPUでも可）
- Flask + flask-cors
- whisper (openai/whisper)
- sudachipy + sudachidict_core
- numpy, etc.
- React (Next.js) フロントエンド（whisper_cab/app）

## セットアップ手順（Ubuntuサーバー）

### 1. 必要なシステムパッケージのインストール

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip build-essential ffmpeg git -y
```

### 2. 仮想環境の作成・有効化

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Pythonパッケージのインストール

```bash
pip install --upgrade pip
pip install git+https://github.com/openai/whisper.git flask flask-cors numpy sudachipy sudachidict_core
```

### 4. CUDA対応PyTorchのインストール（GPU利用時のみ）

PyTorch公式サイト https://pytorch.org/get-started/locally/ でコマンドを確認し、例：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5. SudachiPy辞書のダウンロード（初回のみ）

```bash
python -m sudachipy download
```

### 6. サーバー起動

```bash
cd whisper_meCab_project/server
python main.py
```

- サーバーは http://<サーバーIP>:5000 で起動します。
- `/transcribe` エンドポイントに音声ファイル（wav/m4a等）をPOSTすると、進捗付きで文字起こし・句読点付与結果が返ります。
- 複数リクエストはキューで順番待ちし、フロントで待機状況も表示されます。

### 7. フロントエンド（Next.js）

別途 `whisper_meCab_project/whisper_cab/app` ディレクトリで `npm install` → `npm run build` → `npm start` で本番運用可能。

## ディレクトリ構成

- `server/main.py` : Flask + Whisper + SudachiPy サーバー本体
- `whisper_cab/app/page.js` : Next.jsフロントエンド
- `README.md` : このファイル

## 注意事項

- Whisperモデルの初回ダウンロード・ロードには時間がかかります。
- CUDA非対応環境では `device="cpu"` を指定してください。
- SudachiPyの辞書は `sudachidict_core` を利用しています。
- ffmpegが必要です（aptでインストール済み推奨）。

## ライセンス

本プロジェクトはオープンソースです。商用・非商用問わずご利用いただけます。
