from setuptools import setup, find_packages

setup(
    name="rag-app",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.30.0",
        "fastapi>=0.103.1",
        "uvicorn>=0.23.2",
        "python-dotenv>=1.0.0",
        "pinecone-client>=2.2.2",
        "openai>=0.28.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "tqdm>=4.66.0",
        "nltk>=3.8.0",
        "matplotlib>=3.8.0",
    ],
    python_requires=">=3.9",
)