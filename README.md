# HE-Color-Deconvolution-High-Fidelity-Stain-Separation
基于颜色反卷积的病理图像苏木素(Hematoxylin)提取工具 这是一个用于对 H&amp;E（苏木素-伊红）染色病理切片进行颜色反卷积Color Deconvolution) 的 Python 实现。  本项目重点解决了目前开源社区中常见的误区，区分了 免疫组化模式 (Method A) 与 纯 H&amp;E 模式 (Method B) 的差异，并提供了适合深度学习细胞分割的高保真归一化策略。
## Visual Abstract (效果展示) 

<img width="4500" height="1800" alt="test1_he_deconv_comparison" src="https://github.com/user-attachments/assets/edff5117-4a0f-4e7a-8076-2f98c69a74f2" />
<img width="4500" height="1800" alt="test2_he_deconv_comparison" src="https://github.com/user-attachments/assets/84af4e55-2531-4b4c-9dd4-7f130ef3a6cf" />

> **图示说明**:
> *   **Left**: 原始 H&E RGB 图像。
> *   **Middle (Method A)**: 传统 ImageJ 插件使用的 DAB 模式。注意在无棕色染料的情况下，强制分离会导致苏木素通道不纯。
> *   **Right (Method B )**: 使用正交向量计算的纯 H&E 模式。背景噪音更低，细胞核轮廓与纹理保留更完整。

---

## 🛠️ Installation (安装)

本项目依赖以下基础科学计算库：

```bash
pip install numpy opencv-python matplotlib
```

---

## 🚀 Quick Start (快速开始)

```python
import cv2
from deconvolution import deconv_he, plot_comparison

# 1. 读取图片
img_path = 'sample.png'
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. 执行反卷积 (推荐使用 Method B + High Fidelity)
# 返回结果为 0.0-1.0 的 float32 图像
he_channel = deconv_he(image, matrix_method='B', norm_method='high_fidelity', gamma=1.0)

# 3. 快速生成对比图并保存
plot_comparison(img_path, save_path='result.png')
```

---

## 🧠 Core Theory (核心原理)

### 1. 基础公式
基于 **Ruifrok and Johnston (2001)** 提出的理论，RGB 图像的光强与染色剂浓度呈非线性关系。分离过程需先根据 **比尔-朗伯定律 (Beer-Lambert law)** 将图像转换到 **光密度空间 (Optical Density, OD)**：

$$ OD = -\log_{10}(\frac{I}{I_0}) $$

其中 $I$ 为像素强度，$I_0$ 为入射光强度（通常取 255）。随后通过染色矩阵的逆运算分离通道：

$$ C = OD \cdot M^{-1} $$

### 2. 染色矩阵的两大流派 (Method A vs B)

这是本项目与普通教程最大的不同点。虽然 H（苏木素）和 E（伊红）的向量通常固定，但 **第三行（Residual）** 的定义决定了分离效果。

| 特性 | Method A: H&E + DAB | Method B: H&E + Orthogonal (🔥推荐) |
| :--- | :--- | :--- |
| **定义** | 第三行固定为 **DAB (棕色)** 吸收系数 | 第三行由 **H和E的叉积 (正交向量)** 计算得出 |
| **适用场景** | **免疫组化 (IHC)** 切片 | **纯 H&E** 染色切片 |
| **原理缺陷** | 强制在无棕色区域寻找棕色分量，干扰分离 | 数学上分离最彻底，第三通道仅吸收无法解释的噪音 |
| **典型应用** | 检测特定抗原表达 | 细胞核分割 (Cell Segmentation)、组织分型 |

### 3. 后处理归一化 (Normalization)

分离出的 OD 密度图范围不一，为了适配神经网络输入，提供了两种策略：

*   **`robust` (暴力截断)**:
    *   将 99% 分位点作为最大值进行拉伸。
    *   **效果**: 黑白分明，轮廓极清，但丢失内部纹理。
*   **`high_fidelity` (高保真)**:
    *   使用真实最大值归一化，配合 Gamma 校正。
    *   **效果**: 保留细胞核内部深浅不一的染色细节。

---

## 📂 Project Structure

```text
.
├── deconvolution.py    # 核心算法实现
├── README.md           # 说明文档
└── dataset/            # 存放测试图片
```

## 📝 Reference

*   **Original Paper**: A. C. Ruifrok and D. A. Johnston, *"Quantification of histochemical staining by color deconvolution,"* Analytical and Quantitative Cytology and Histology, 2001.

---

## 🤝 Contributing

欢迎提交 Issue 或 Pull Request。

---
*Created by [Yunbo Gong]*
