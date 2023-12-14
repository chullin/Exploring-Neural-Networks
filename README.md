# Exploring-Neural-Networks
Exploring Neural Networks: Playful Adventures with PyTorch and TensorFlow


以下是在使用 GPU 的情況下安裝 TensorFlow 並設置 Python 虛擬環境的一般步驟。請確保您已經安裝了適當的 GPU 驅動程序，並且您的 GPU 支持 CUDA（如果您想要使用 GPU 運行 TensorFlow）。

## 步驟一：安裝 NVIDIA CUDA Toolkit 和 cuDNN
1. 安裝 NVIDIA CUDA Toolkit：
    * 根據您的 GPU 和操作系統版本，從 NVIDIA 官方網站下載並安裝 CUDA Toolkit：https://developer.nvidia.com/cuda-toolkit

    ![Alt text](image.png)
2. 安裝 cuDNN：
    * 從 NVIDIA 官方網站下載 cuDNN：https://developer.nvidia.com/cudnn
    * 將 cuDNN 解壓縮，將文件複製到 CUDA Toolkit 的安裝目錄中。

## 步驟二：創建 Python 虛擬環境
1. 安裝虛擬環境管理工具（如果未安裝）：

```bash
pip install virtualenv
```

2. 創建虛擬環境：

```bash
virtualenv myenv
```
3. 啟動虛擬環境（Linux/Mac）：

```bash
source myenv/bin/activate
```
啟動虛擬環境（Windows）：

```bash
.\myenv\Scripts\activate
```

## 步驟三：安裝 TensorFlow
1. 安裝 TensorFlow：
    * 安裝 CPU 版本（如果沒有 GPU）：

```bash
pip install tensorflow
```

2. 安裝 GPU 版本（在已經安裝 CUDA 和 cuDNN 的情況下）：

```bash
pip install tensorflow-gpu
```

## 步驟四：驗證 TensorFlow 安裝
1. 在 Python 環境中驗證 TensorFlow 安裝：

```python
Copy code
import tensorflow as tf
print(tf.__version__)
```
如果一切順利，您應該看到 TensorFlow 的版本信息。

### 注意事項：
* 如果您使用 Anaconda，可以使用 conda 來創建和管理虛擬環境，但仍然需要通過 pip 安裝 TensorFlow。
* 在 Windows 上，可能需要安裝 Microsoft Visual C++ Redistributable，以便在編譯期間解決一些依賴項。

請注意，這只是一般的步驟，實際操作中可能會因系統配置和軟件版本的不同而有所不同。確保參考 TensorFlow 和 CUDA/cuDNN 的官方文檔以獲取最準確的信息。