# DBNet++ Implementation Guide

DBNet++（Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion）の実装と詳細解説。

## 概要

DBNet++は高精度かつリアルタイムなテキスト検出を実現するディープラーニングモデルです。主な革新点は：

1. **Differentiable Binarization (DB)**: 微分可能な二値化処理
2. **Adaptive Scale Fusion (ASF)**: 適応的なマルチスケール特徴統合
3. **Multi-scale Architecture**: バックボーン + FPN + 注意機構

## アーキテクチャ詳細

### 1. ResNet Backbone

**役割**: 画像から多段階の特徴量を抽出

```python
# 入力: (1, 3, 640, 640)
# 出力:
c2: (1, 64, 160, 160)   # 1/4 scale - 細かいテキスト詳細
c3: (1, 128, 80, 80)    # 1/8 scale - 中程度のテキスト
c4: (1, 256, 40, 40)    # 1/16 scale - 大きめのテキスト
c5: (1, 512, 20, 20)    # 1/32 scale - 最も抽象的な文脈情報
```

**BasicBlockの構造**:
- Residual Connection（残差接続）により深いネットワークでも安定学習
- `output = F(input) + input` の形で勾配消失を防止

### 2. Feature Pyramid Network (FPN)

**役割**: 異なる解像度の特徴量を統合し、マルチスケール情報を活用

#### 処理フロー

1. **Lateral Connections**: 各レベルの特徴を統一チャンネル数（256）に変換
2. **Top-down Path**: 高レベル特徴を低レベルに向けて順次融合
3. **Output Smoothing**: 3x3畳み込みでエイリアシングを除去

```python
# FPN出力: 全て256チャンネル、元の解像度維持
P2: (1, 256, 160, 160)  # 小さなテキスト検出用
P3: (1, 256, 80, 80)    # 中程度のテキスト検出用
P4: (1, 256, 40, 40)    # 大きなテキスト検出用
P5: (1, 256, 20, 20)    # グローバル文脈用
```

**FPN vs U-Net**:
- FPN: 各レベルで独立出力、オブジェクト検出に最適
- U-Net: 単一の最終出力、セグメンテーションに最適

### 3. Adaptive Scale Fusion (ASF) - DBNet++の改良点

**役割**: FPNの4つのスケール特徴を適応的に統合

#### 2段階注意機構

**Stage 1: Stage-wise Attention**
```python
# グローバル特徴から「どのスケールが重要か」を学習
stage_attention = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),    # (H,W) → (1,1)
    nn.Conv2d(1024, 256, 1),    # チャンネル削減
    nn.Conv2d(256, 1024, 1),    # 復元
    nn.Sigmoid()                # 0-1の重み
)
```

**Stage 2: Spatial Attention**
```python
# 「どの空間位置が重要か」を学習
spatial_attention = nn.Sequential(
    nn.Conv2d(1024, 256, 1),           # チャンネル削減
    nn.Conv2d(256, 1, 7, padding=3),   # 大きな受容野
    nn.Sigmoid()                       # 0-1の重み
)
```

#### 効果
- **適応的スケール選択**: 小さなテキスト→P2重視、大きなテキスト→P4重視
- **空間適応性**: テキスト領域で高い注意、背景で低い注意

### 4. DBHead - 核心技術

**役割**: テキスト検出のための3つのマップを生成

#### 出力
1. **Probability Map (P)**: テキスト領域の確率 (0-1)
2. **Threshold Map (T)**: 各ピクセルの適応的閾値 (0-1)  
3. **Binary Map (B)**: 最終的な二値化結果 (0 or 1)

#### Differentiable Binarization

**従来の問題**:
```python
binary = (prob > 0.5).float()  # 固定閾値、微分不可能
```

**DBの解決**:
```python
# 訓練時: 微分可能なシグモイド近似
binary = 1 / (1 + exp(-k * (prob - thresh)))

# 推論時: 通常のステップ関数  
binary = (prob >= thresh).float()
```

**革新性**:
- 各ピクセルが独自の適応的閾値を持つ
- 閾値マップも学習により最適化される
- 二値化処理を学習に統合可能

## 損失関数設計

### 3つの損失の組み合わせ

```python
total_loss = α × L_prob + β × L_binary + γ × L_thresh
```

### 各損失の詳細

#### 1. Probability & Binary Map Loss
```python
# Dice Loss: 領域の重なり度合いを直接最適化
dice_loss = 1 - (2 × intersection + ε) / (union + ε)

# Balanced BCE: クラス不均衡に対応
# Online Hard Example Mining (OHEM)で困難な負例のみ学習
```

#### 2. Threshold Map Loss（革新的部分）
```python
# 境界付近で適応的閾値を学習
threshold_target = distance_inside / max_distance  # 内部ほど高い値
boundary_mask = (distance_to_boundary <= 3)        # 境界付近のみ
```

### 重み設定の意味
```python
# 典型的設定: α=1.0, β=10.0, γ=1.0
# β=10.0が大きい理由: Binary mapが最終出力で最重要
```

## データフロー全体

```python
# 1. 入力画像
image: (B, 3, H, W)

# 2. バックボーン特徴抽出
backbone_features = {
    'c2': (B, 64, H/4, W/4),
    'c3': (B, 128, H/8, W/8), 
    'c4': (B, 256, H/16, W/16),
    'c5': (B, 512, H/32, W/32)
}

# 3. FPN統合
fpn_features = [
    (B, 256, H/4, W/4),   # P2
    (B, 256, H/8, W/8),   # P3
    (B, 256, H/16, W/16), # P4
    (B, 256, H/32, W/32)  # P5
]

# 4. ASF統合
fused_features: (B, 1024, H/4, W/4)  # 4スケール×256チャンネル

# 5. DBHead出力
outputs = {
    'prob_map': (B, 1, H, W),     # 確率マップ
    'thresh_map': (B, 1, H, W),   # 閾値マップ  
    'binary_map': (B, 1, H, W)    # 二値化マップ
}
```

## 学習の特徴

### クラス不均衡への対処
- **問題**: テキスト領域5% vs 背景領域95%
- **解決**: Dice Loss + OHEM + 重み付けBCE

### 段階的学習進行
```
初期段階: prob_loss↑, binary_loss↑, thresh_loss↑
収束段階: prob_loss↓, binary_loss↓, thresh_loss↓
```

## 主要な革新点

1. **適応的二値化**: 固定閾値→各ピクセル独自閾値
2. **微分可能性**: 二値化処理を学習に統合
3. **マルチスケール統合**: ASFによる適応的特徴融合
4. **複合損失**: 3つの異なる観点からの最適化

## 性能

- **精度**: F-measure 82.8%（MSRA-TD500）
- **速度**: 62 FPS（ResNet-18 backbone）
- **効率性**: 精度と速度の理想的なトレードオフを実現

## 実装ファイル

- `src/cvt/dbnet.py`: モデル実装（Backbone, FPN, ASF, DBHead）
- `src/cvt/dbnet_loss.py`: 損失関数実装
- 今後実装予定: 訓練ループ、評価メトリクス、可視化ツール

---

このドキュメントはDBNet++の理論と実装の詳細な解説です。実際のコードと合わせて参照してください。