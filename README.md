# PIP多模态图像助理

![微信截图_20241102233935](https://github.com/user-attachments/assets/e2746ba5-cbd9-4dec-8528-c1f384d1ec8b)

![微信截图_20241102233005](https://github.com/user-attachments/assets/e0f09115-cfc5-4eef-be1a-f6006956ea52)

PIP多模态图像助理是一个基于Janus魔改的多模态理解的图像处理工具，你可以用这个方法询问任何关于图像的信息，同时支持批量标注图像。
本来想自用的后来发现这个模型现在准确性一般，但优点是他很小。
基本上你可以问关于你上传的图像的任何内容，我提供了一个批量询问的方法（你可以在问询模板.json中找到并编辑它）。会把你所有询问问题的结果储存到图像路径下的json里。

## 安装

请按照以下步骤安装和运行项目：

1.克隆到本地
git clone https://github.com/chenpipi0807/PIP_Multimodal-image-query-assistant.git

2.使用powershell 运行这玩意补全模型和其他乱七八糟的玩意：  download.ps1

3.不出意外就用这玩意运行就行了： RUN_GUI.py


## 功能

- **单张图像处理**：上传一张图片并提出一个问题以获取描述。
- **批量图像处理**：批量处理图像并根据输入的问题生成结果。

## 注意事项

- 确保您的环境中安装了Git LFS以下载大模型文件。
- 如果您在中文路径下运行，请确保您的Python和相关工具支持中文路径。
