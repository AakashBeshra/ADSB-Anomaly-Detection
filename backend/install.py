"""
Installation script with Python 3.14 compatibility
"""
import subprocess
import sys
import os

def install_with_compat():
    """Install packages with compatibility workarounds"""
    
    # First install packages that don't need build
    print("📦 Installing base packages...")
    
    packages = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "pydantic==2.5.0",
        "sqlalchemy==2.0.23",
        "aiosqlite==0.19.0",
        "websockets==12.0",
        "aiofiles==23.2.1",
        "python-multipart==0.0.6",
        "python-dotenv==1.0.0",
        "redis==5.0.1",
        "joblib==1.3.2",
        "pydantic-settings==2.1.0",
        "httpx==0.25.2",
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Install numpy with specific options
    print("📦 Installing numpy (with compatibility workaround)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "numpy==1.26.4",
        "--no-build-isolation",
        "--only-binary=:all:"
    ])
    
    # Install pandas
    print("📦 Installing pandas...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "pandas==2.2.0",
        "--no-build-isolation"
    ])
    
    # Install scikit-learn
    print("📦 Installing scikit-learn...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "scikit-learn==1.4.0",
        "--no-build-isolation"
    ])
    
    # Install tensorflow
    print("📦 Installing tensorflow...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "tensorflow==2.15.0",
        "--no-deps"
    ])
    
    print("\n✅ Installation complete!")

if __name__ == "__main__":
    print(f"🐍 Python version: {sys.version}")
    
    if sys.version_info >= (3, 14):
        print("⚠️  Python 3.14+ detected - using compatibility installation")
        # Apply environment variable for compatibility
        os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    
    install_with_compat()