<div align="center">

# 🌊 PoseWave

**WiFi CSI Human Pose Detection System**

*Privacy-preserving pose detection using WiFi signals — No cameras required!*

[![GitHub Stars](https://img.shields.io/github/stars/gitstq/PoseWave?style=for-the-badge&logo=github&color=yellow)](https://github.com/gitstq/PoseWave/stargazers)
[![GitHub License](https://img.shields.io/github/license/gitstq/PoseWave?style=for-the-badge&color=blue)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-green?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)

[English](#english) | [简体中文](#简体中文) | [繁體中文](#繁體中文)

</div>

---

<a name="english"></a>
## 📖 English

### 🎉 Project Introduction

**PoseWave** is an innovative **WiFi CSI (Channel State Information) based human pose detection system** that enables real-time human pose recognition without any cameras or visual sensors. By analyzing WiFi signal variations caused by human body movements, PoseWave achieves privacy-preserving, non-intrusive pose detection ideal for smart homes, healthcare monitoring, and security applications.

**💡 Key Differentiators:**
- 🔒 **100% Privacy-Preserving**: No cameras, no visual data, complete privacy protection
- ⚡ **Real-time Detection**: Millisecond-level response for immediate alerts
- 🏠 **Smart Home Ready**: Fall detection, elderly monitoring, intrusion alerts
- 📡 **WiFi Infrastructure**: Leverages existing WiFi networks, no additional hardware
- 🛠️ **Easy Integration**: RESTful API + WebSocket for seamless integration

### ✨ Core Features

| Feature | Description |
|---------|-------------|
| 🔍 **Pose Detection** | Detect standing, sitting, lying, walking, falling, and empty room |
| 📊 **CSI Processing** | Advanced signal processing with bandpass filtering, phase cleaning |
| 🧠 **Deep Learning** | CNN-LSTM architecture (WavePoseNet) for accurate classification |
| 📡 **Real-time API** | FastAPI-based REST API with WebSocket streaming |
| 🔔 **Alert System** | Automatic fall detection with instant alerts |
| 📈 **Statistics** | Detection history, confidence tracking, pose distribution |

### 🚀 Quick Start

#### Prerequisites
- Python 3.9+
- pip package manager

#### Installation

```bash
# Clone the repository
git clone https://github.com/gitstq/PoseWave.git
cd PoseWave

# Install dependencies
pip install -r requirements.txt

# Or install with development tools
pip install -e ".[dev]"
```

#### Run the API Server

```bash
# Start the server
uvicorn posewave.api.main:app --host 0.0.0.0 --port 8000

# Or use the Makefile
make run
```

#### API Documentation

Once running, access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 📖 Detailed Usage Guide

#### Basic Usage

```python
from posewave.core.csi_processor import CSIProcessor
from posewave.core.pose_detector import PoseDetector
import numpy as np

# Initialize components
processor = CSIProcessor()
detector = PoseDetector()

# Process CSI data (simulated)
csi_data = np.random.randn(64, 3, 100)  # 64 subcarriers, 3 antennas, 100 samples
features = processor.process(csi_data)

# Detect pose
result = detector.detect(features)

print(f"Detected Pose: {result.pose_type.value}")
print(f"Confidence: {result.confidence:.2%}")
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/detect` | POST | Detect pose from CSI data |
| `/api/v1/features` | POST | Extract features without detection |
| `/api/v1/statistics` | GET | Get detection statistics |
| `/api/v1/poses` | GET | List supported poses |
| `/api/v1/simulate` | GET | Simulate detection with synthetic data |

#### Example API Call

```bash
# Detect pose
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Content-Type: application/json" \
  -d '{"data": [[[1.0, 2.0], [3.0, 4.0]]], "sample_rate": 100.0}'
```

### 💡 Design Philosophy

**Why WiFi CSI for Pose Detection?**

Traditional pose detection relies on cameras or wearable sensors, which have limitations:
- 🚫 Cameras raise privacy concerns
- 🚫 Wearables require user compliance
- 🚫 Specialized sensors are expensive

WiFi CSI offers a unique solution:
- ✅ Already deployed in most buildings
- ✅ Works through walls and in darkness
- ✅ No privacy concerns (no visual data)
- ✅ Non-intrusive (no wearables needed)

**Architecture Overview:**

```
WiFi Router → CSI Data → Signal Processing → Feature Extraction → Pose Classification
                                              ↓
                                        Real-time Alerts
```

### 📦 Deployment

#### Docker Deployment

```bash
# Build image
docker build -t posewave:latest .

# Run container
docker run -p 8000:8000 posewave:latest

# Or use docker-compose
docker-compose up -d
```

#### Production Deployment

```bash
# Run with multiple workers
uvicorn posewave.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<a name="简体中文"></a>
## 📖 简体中文

### 🎉 项目介绍

**PoseWave** 是一个创新的**基于WiFi CSI（信道状态信息）的人体姿态检测系统**，无需任何摄像头或视觉传感器即可实现实时人体姿态识别。通过分析人体运动引起的WiFi信号变化，PoseWave实现了隐私保护、非侵入式的姿态检测，非常适合智能家居、健康监护和安全应用场景。

**💡 自研差异化亮点：**
- 🔒 **100%隐私保护**：无摄像头、无视觉数据，完全保护隐私
- ⚡ **实时检测**：毫秒级响应，即时告警
- 🏠 **智能家居就绪**：跌倒检测、老人监护、入侵告警
- 📡 **WiFi基础设施**：利用现有WiFi网络，无需额外硬件
- 🛠️ **易于集成**：RESTful API + WebSocket，无缝对接

### ✨ 核心特性

| 特性 | 描述 |
|------|------|
| 🔍 **姿态检测** | 检测站立、坐着、躺下、行走、跌倒、无人等状态 |
| 📊 **CSI处理** | 高级信号处理，包括带通滤波、相位清洗 |
| 🧠 **深度学习** | CNN-LSTM架构（WavePoseNet）实现精准分类 |
| 📡 **实时API** | 基于FastAPI的REST API，支持WebSocket流式传输 |
| 🔔 **告警系统** | 自动跌倒检测，即时告警 |
| 📈 **统计分析** | 检测历史、置信度追踪、姿态分布统计 |

### 🚀 快速开始

#### 环境要求
- Python 3.9+
- pip 包管理器

#### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/gitstq/PoseWave.git
cd PoseWave

# 安装依赖
pip install -r requirements.txt

# 或安装开发版本
pip install -e ".[dev]"
```

#### 启动API服务

```bash
# 启动服务器
uvicorn posewave.api.main:app --host 0.0.0.0 --port 8000

# 或使用Makefile
make run
```

#### API文档

启动后访问交互式API文档：
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 📖 详细使用指南

#### 基础用法

```python
from posewave.core.csi_processor import CSIProcessor
from posewave.core.pose_detector import PoseDetector
import numpy as np

# 初始化组件
processor = CSIProcessor()
detector = PoseDetector()

# 处理CSI数据（模拟数据）
csi_data = np.random.randn(64, 3, 100)  # 64子载波，3天线，100采样点
features = processor.process(csi_data)

# 检测姿态
result = detector.detect(features)

print(f"检测姿态: {result.pose_type.value}")
print(f"置信度: {result.confidence:.2%}")
```

#### API接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/v1/detect` | POST | 从CSI数据检测姿态 |
| `/api/v1/features` | POST | 提取特征（不检测） |
| `/api/v1/statistics` | GET | 获取检测统计 |
| `/api/v1/poses` | GET | 列出支持的姿态 |
| `/api/v1/simulate` | GET | 使用合成数据模拟检测 |

### 💡 设计思路

**为什么选择WiFi CSI进行姿态检测？**

传统姿态检测依赖摄像头或可穿戴传感器，存在以下局限：
- 🚫 摄像头引发隐私担忧
- 🚫 可穿戴设备需要用户配合
- 🚫 专业传感器成本高昂

WiFi CSI提供了独特解决方案：
- ✅ 大多数建筑已部署WiFi
- ✅ 可穿墙工作，无光照要求
- ✅ 无隐私担忧（无视觉数据）
- ✅ 非侵入式（无需穿戴设备）

**架构概览：**

```
WiFi路由器 → CSI数据 → 信号处理 → 特征提取 → 姿态分类
                                    ↓
                              实时告警
```

### 📦 打包与部署

#### Docker部署

```bash
# 构建镜像
docker build -t posewave:latest .

# 运行容器
docker run -p 8000:8000 posewave:latest

# 或使用docker-compose
docker-compose up -d
```

#### 生产环境部署

```bash
# 多worker运行
uvicorn posewave.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

### 📄 开源协议

本项目采用MIT协议开源 - 详见 [LICENSE](LICENSE) 文件。

---

<a name="繁體中文"></a>
## 📖 繁體中文

### 🎉 專案介紹

**PoseWave** 是一個創新的**基於WiFi CSI（通道狀態資訊）的人體姿態檢測系統**，無需任何攝影機或視覺感測器即可實現即時人體姿態辨識。透過分析人體運動引起的WiFi訊號變化，PoseWave實現了隱私保護、非侵入式的姿態檢測，非常適合智慧家庭、健康照護和安全應用場景。

**💡 自研差異化亮點：**
- 🔒 **100%隱私保護**：無攝影機、無視覺資料，完全保護隱私
- ⚡ **即時檢測**：毫秒級響應，即時告警
- 🏠 **智慧家庭就緒**：跌倒檢測、長者照護、入侵告警
- 📡 **WiFi基礎設施**：利用現有WiFi網路，無需額外硬體
- 🛠️ **易於整合**：RESTful API + WebSocket，無縫對接

### ✨ 核心特性

| 特性 | 描述 |
|------|------|
| 🔍 **姿態檢測** | 檢測站立、坐著、躺下、行走、跌倒、無人等狀態 |
| 📊 **CSI處理** | 進階訊號處理，包括帶通濾波、相位清洗 |
| 🧠 **深度學習** | CNN-LSTM架構（WavePoseNet）實現精準分類 |
| 📡 **即時API** | 基於FastAPI的REST API，支援WebSocket串流傳輸 |
| 🔔 **告警系統** | 自動跌倒檢測，即時告警 |
| 📈 **統計分析** | 檢測歷史、置信度追蹤、姿態分佈統計 |

### 🚀 快速開始

#### 環境要求
- Python 3.9+
- pip 套件管理器

#### 安裝步驟

```bash
# 複製儲存庫
git clone https://github.com/gitstq/PoseWave.git
cd PoseWave

# 安裝相依套件
pip install -r requirements.txt

# 或安裝開發版本
pip install -e ".[dev]"
```

#### 啟動API服務

```bash
# 啟動伺服器
uvicorn posewave.api.main:app --host 0.0.0.0 --port 8000

# 或使用Makefile
make run
```

#### API文件

啟動後存取互動式API文件：
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 📖 詳細使用指南

#### 基礎用法

```python
from posewave.core.csi_processor import CSIProcessor
from posewave.core.pose_detector import PoseDetector
import numpy as np

# 初始化元件
processor = CSIProcessor()
detector = PoseDetector()

# 處理CSI資料（模擬資料）
csi_data = np.random.randn(64, 3, 100)  # 64子載波，3天線，100取樣點
features = processor.process(csi_data)

# 檢測姿態
result = detector.detect(features)

print(f"檢測姿態: {result.pose_type.value}")
print(f"置信度: {result.confidence:.2%}")
```

### 💡 設計思路

**為什麼選擇WiFi CSI進行姿態檢測？**

傳統姿態檢測依賴攝影機或可穿戴感測器，存在以下限制：
- 🚫 攝影機引發隱私擔憂
- 🚫 可穿戴裝置需要使用者配合
- 🚫 專業感測器成本高昂

WiFi CSI提供了獨特解決方案：
- ✅ 大多數建築已部署WiFi
- ✅ 可穿牆運作，無光照要求
- ✅ 無隱私擔憂（無視覺資料）
- ✅ 非侵入式（無需穿戴裝置）

### 📦 打包與部署

#### Docker部署

```bash
# 建構映像檔
docker build -t posewave:latest .

# 執行容器
docker run -p 8000:8000 posewave:latest

# 或使用docker-compose
docker-compose up -d
```

### 🤝 貢獻指南

歡迎貢獻程式碼！請遵循以下步驟：

1. Fork本儲存庫
2. 建立功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交變更 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 建立Pull Request

### 📄 開源授權

本專案採用MIT授權條款開源 - 詳見 [LICENSE](LICENSE) 檔案。

---

<div align="center">

**⭐ If you find PoseWave useful, please give it a star! ⭐**

**⭐ 如果您觉得PoseWave有用，请给个星标支持！⭐**

**⭐ 如果您覺得PoseWave有用，請給個星標支持！⭐**

Made with ❤️ by PoseWave Team

</div>
