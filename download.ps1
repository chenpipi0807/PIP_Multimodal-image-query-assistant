# 创建虚拟环境
if (-Not (Test-Path -Path ".\venv")) {
    python -m venv venv
}

# 激活虚拟环境
& .\venv\Scripts\Activate

# 克隆Janus项目
if (-Not (Test-Path -Path ".\Janus")) {
    git clone https://github.com/deepseek-ai/Janus.git
}

# 进入Janus目录
Set-Location -Path .\Janus

# 安装项目依赖
pip install -e .
pip install -r requirements.txt

# 返回上级目录
Set-Location -Path ..

# 创建models文件夹
if (-Not (Test-Path -Path ".\models")) {
    New-Item -ItemType Directory -Name models -Force
}

# 进入models目录
Set-Location -Path .\models

# 安装Git LFS
git lfs install

# 克隆模型
if (-Not (Test-Path -Path ".\Janus-1.3B")) {
    git lfs clone https://huggingface.co/deepseek-ai/Janus-1.3B.git
}

# 返回上级目录
Set-Location -Path ..

# 执行 RUN_GUI.py
try {
    python .\Janus\RUN_GUI.py
} catch {
    Write-Host "执行失败，请检查错误信息。"
}

# 暂停以保持窗口打开
Read-Host -Prompt "按任意键退出"
