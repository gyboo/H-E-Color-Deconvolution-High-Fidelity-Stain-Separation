import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_deconvolution_matrix(method='B'):
    """
    根据流派生成反卷积矩阵（逆矩阵）。
    """
    # 标准 H&E 吸收系数 (Ruifrok & Johnston)
    # 归一化处理以保证数学严谨性
    H = np.array([0.644211, 0.716556, 0.266844])
    E = np.array([0.092789, 0.954111, 0.283111])
    
    # # 确保向量归一化
    H = H / np.linalg.norm(H)
    E = E / np.linalg.norm(E)

    if method == 'A':
        # --- 流派 A: H + E + DAB (免疫组化模式) ---
        # 第三通道固定为 DAB (棕色)
        DAB = np.array([0.635954, 0.001000, 0.771720])
        DAB = DAB / np.linalg.norm(DAB)
        stain_matrix = np.array([H, E, DAB])
        
    elif method == 'B':
        # --- 流派 B: H + E + Orthogonal (纯 H&E 模式 ) ---
        # 第三通道为计算出的正交向量 (既不是H也不是E的残差)
        Res = np.cross(H, E)
        Res = Res / np.linalg.norm(Res)
        stain_matrix = np.array([H, E, Res])
        
    else:
        raise ValueError("Method must be 'A' or 'B'")

    # 返回逆矩阵
    return np.linalg.inv(stain_matrix)


def deconv_he(image, matrix_method='B', norm_method='high_fidelity', gamma=1.0):
    """
    执行颜色反卷积并提取苏木素(H)通道。

    Args:
        image: 输入图像 (H, W, 3), RGB, range [0, 255]
        matrix_method: 'A' (DAB模式) 或 'B' (正交模式, 默认推荐)
        norm_method: 
            'robust' - 99%截断，黑白分明
            'high_fidelity' - 保留纹理细节
        gamma: Gamma校正值。默认 1.0 (不使用)。
               推荐 0.6 用于增强细胞边缘细节。

    Returns:
        processed_img: (H, W, 3), range [0.0, 1.0], float32
    """
    # 1. 预处理：转浮点，防止 log(0)
    img_float = image.astype(np.float32) + 1e-6
    
    # 2. 转换到光密度空间 (OD)
    od = -np.log10(img_float / 255.0)
    
    # 3. 获取反卷积矩阵并运算
    deconv_matrix = get_deconvolution_matrix(method=matrix_method)
    print(f"使用反卷积矩阵 (Method {matrix_method}):\n{deconv_matrix}")
    h, w, c = image.shape

    print(deconv_matrix.T.shape)
    # 矩阵乘法: (Pixel_N, 3) @ (3, 3).T
    stain_densities = np.dot(od.reshape((-1, 3)), deconv_matrix).reshape((h, w, 3))
    
    # 4. 提取苏木素 (Channel 0) 并清理噪音
    h_density = stain_densities[:, :, 0]

    h_density = np.maximum(h_density, 0) # 截断负值噪音

    # 5. 归一化策略
    if norm_method == 'robust':
        # 暴力拉伸：忽略最亮 1%
        min_val = np.min(h_density)
        max_val = np.percentile(h_density, 99.0)
        h_norm = (h_density - min_val) / (max_val - min_val + 1e-8)
        
    elif norm_method == 'high_fidelity':
        # 高保真：保留最大值
        min_val = np.min(h_density)
        max_val = np.max(h_density)
        h_norm = (h_density - min_val) / (max_val - min_val + 1e-8)
        
    else:
        h_norm = h_density # 不归一化 (不推荐)

    h_norm = np.power(h_norm, gamma)
    # 7. 堆叠输出 (H, W, 3)
    h_norm = np.clip(h_norm, 0, 1)
    return np.stack([h_norm, h_norm, h_norm], axis=2).astype(np.float32)


def plot_comparison(img_path, save_path=None):
    """
    可视化函数：读取图片，生成 原图 + 流派A + 流派B 的对比图
    """
    # 读取图片 (OpenCV默认是BGR，需转RGB)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Error: 无法读取图片 {img_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. 计算流派 A (DAB)
    res_A = deconv_he(img_rgb, matrix_method='A', norm_method='high_fidelity', gamma=1.0)

    # 2. 计算流派 B (Orthogonal)
    res_B = deconv_he(img_rgb, matrix_method='B', norm_method='high_fidelity', gamma=1.0)
    # --- 绘图 ---
    plt.figure(figsize=(15, 6))

    # 子图 1: 原图
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original H&E (RGB)", fontsize=14)
    plt.axis('off')

    # 子图 2: 流派 A
    plt.subplot(1, 3, 2)
    plt.imshow(res_A[:, :, 0], cmap='gray') # 显示单通道灰度
    plt.title("Method A: H+E+DAB\n(IHC Preset)", fontsize=14)
    plt.axis('off')

    # 子图 3: 流派 B
    plt.subplot(1, 3, 3)
    plt.imshow(res_B[:, :, 0], cmap='gray') # 显示单通道灰度
    plt.title("Method B: H+E+Orthogonal\n(Standard H&E Preset)", fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"对比图已保存至: {save_path}")
    
    # plt.show()

if __name__ == "__main__":
    # 请替换为你的本地图片路径
    # input_image_path = r'test1.png' 
    input_image_path = r'dataset/test2.png' 
    # 运行对比并显示
    plot_comparison(input_image_path, save_path='dataset/test2_he_deconv_comparison.png')