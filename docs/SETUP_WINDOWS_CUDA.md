# MiraTTS Windows + CUDA 環境構築手順

このドキュメントは、Windows環境でMiraTTSをCUDA対応で動作させるための環境構築手順をまとめたものです。

## 環境情報

| 項目 | バージョン |
|------|-----------|
| OS | Windows 11 |
| CUDA | 12.x (v13.0も動作確認済み) |
| Python | 3.11 |
| パッケージマネージャ | uv |
| GPU VRAM | 6GB以上推奨 |

## 発生した問題と解決策

### 問題1: nvidia-nccl-cu12がWindows非対応

#### 症状
```
error: Distribution `nvidia-nccl-cu12==2.28.9` can't be installed because it doesn't have a source distribution or wheel for the current platform

hint: You're on Windows (`win_amd64`), but `nvidia-nccl-cu12` only has wheels for: `manylinux_2_18_aarch64`, `manylinux_2_18_x86_64`
```

#### 原因
`nvidia-nccl-cu12`はLinux専用パッケージで、Windowsには対応していない。lmdeployの依存関係として要求される。

#### 解決策
`pyproject.toml`に以下を追加し、Linuxのみでインストールするよう制限：

```toml
[tool.uv]
override-dependencies = [
    "nvidia-nccl-cu12 ; sys_platform == 'linux'",
]
```

### 問題2: PyTorchがCPU版でインストールされる

#### 症状
```
AssertionError: Torch not compiled with CUDA enabled
```

確認方法：
```bash
uv run python -c "import torch; print(torch.__version__)"
# 出力: 2.8.0+cpu  ← CPU版
```

#### 原因
PyPIからインストールされるtorchはデフォルトでCPU版。

#### 解決策
PyTorch公式のCUDA 12.4インデックスを明示的に指定：

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cu124", marker = "sys_platform == 'win32'" },
    { index = "pytorch-cu124", marker = "sys_platform == 'linux'" },
]
torchaudio = [
    { index = "pytorch-cu124", marker = "sys_platform == 'win32'" },
    { index = "pytorch-cu124", marker = "sys_platform == 'linux'" },
]
torchvision = [
    { index = "pytorch-cu124", marker = "sys_platform == 'win32'" },
    { index = "pytorch-cu124", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

また、`dependencies`にtorch関連を明示的に追加：

```toml
dependencies = [
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "torchvision>=0.15.0",
    # ... 他の依存関係
]
```

### 問題3: 不足依存関係

#### 症状
```
ModuleNotFoundError: No module named 'torchaudio'
ModuleNotFoundError: No module named 'omegaconf'
```

#### 解決策
`pyproject.toml`の`dependencies`に追加：

```toml
dependencies = [
    # ... 既存の依存関係
    "torchaudio>=2.0.0",
    "omegaconf>=2.3.0",
]
```

## 最終的なpyproject.toml設定

```toml
[project]
dependencies = [
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "torchvision>=0.15.0",
    "lmdeploy",
    "librosa",
    "fastaudiosr @ git+https://github.com/ysharma3501/FlashSR.git",
    "ncodec @ git+https://github.com/ysharma3501/FastBiCodec.git",
    "einops",
    "onnxruntime-gpu",
    "omegaconf>=2.3.0",
]

[tool.uv]
override-dependencies = [
    "nvidia-nccl-cu12 ; sys_platform == 'linux'",
]

[tool.uv.sources]
torch = [
    { index = "pytorch-cu124", marker = "sys_platform == 'win32'" },
    { index = "pytorch-cu124", marker = "sys_platform == 'linux'" },
]
torchaudio = [
    { index = "pytorch-cu124", marker = "sys_platform == 'win32'" },
    { index = "pytorch-cu124", marker = "sys_platform == 'linux'" },
]
torchvision = [
    { index = "pytorch-cu124", marker = "sys_platform == 'win32'" },
    { index = "pytorch-cu124", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

## インストール手順

### 1. リポジトリをクローン

```bash
git clone https://github.com/ayutaz/MiraTTS.git
cd MiraTTS
```

### 2. 依存関係をインストール

```bash
uv sync
```

### 3. CUDA対応を確認

```bash
uv run python -c "import torch; print(f'torch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

期待される出力：
```
torch: 2.6.0+cu124
CUDA: True
```

## 動作確認

### 基本的な動作確認

```bash
uv run python -c "
from mira.model import MiraTTS
mira_tts = MiraTTS('YatharthS/MiraTTS')
print('Model loaded successfully!')
"
```

### 音声生成テスト

```bash
uv run python test_speed.py
```

## 速度計測結果

### テスト環境
- GPU: NVIDIA GPU with CUDA 12.x
- VRAM: 6GB+

### 結果

| テスト | 生成時間 | 音声長 | リアルタイム倍率 |
|--------|----------|--------|------------------|
| Test 1 (短文) | 4.726s | 5.300s | 1.1x |
| Test 2 (中文) | 1.502s | 6.060s | 4.0x |
| Test 3 (長文) | 2.178s | 8.700s | 4.0x |
| **平均** | 8.405s | 20.060s | **2.4x** |

### 備考
- 最初のテキスト生成は「ウォームアップ」で遅い（KVキャッシュが空のため）
- 2回目以降はKVキャッシュが効いて高速化
- バッチ処理を使用するとさらに高速化可能

## トラブルシューティング

### uv.lockが古い場合

```bash
rm uv.lock
uv sync
```

### CUDAが認識されない場合

1. CUDAツールキットがインストールされているか確認
2. `nvcc --version`でCUDAバージョンを確認
3. 環境変数`CUDA_HOME`が設定されているか確認

### メモリ不足の場合

`mira/model.py`の`cache_max_entry_count`パラメータを調整：

```python
backend_config = TurbomindEngineConfig(
    cache_max_entry_count=0.1,  # デフォルト0.2から減らす
    # ...
)
```

## 参考リンク

- [uv PyTorch Guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [LMDeploy Installation](https://lmdeploy.readthedocs.io/en/latest/get_started/installation.html)
- [MiraTTS HuggingFace](https://huggingface.co/YatharthS/MiraTTS)
