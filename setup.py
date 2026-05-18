from setuptools import setup, find_packages

setup(
    name="vortexrag",
    version="0.1.0",
    description="Vector Orthogonal Resonance-Tuned EXtraction RAG — kills semantic drift and context poisoning simultaneously",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Vignesh",
    author_email="lkvarnesh@gmail.com",
    url="https://github.com/vignesh2027/VORTEXRAG",
    license="MIT",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        "full": [
            "sentence-transformers>=2.2.0",
            "spacy>=3.5.0",
            "faiss-cpu>=1.7.0",
            "networkx>=3.0",
            "rouge-score>=0.1.2",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
    ],
    keywords="RAG retrieval-augmented-generation semantic-search NLP LLM hallucination",
    project_urls={
        "Documentation": "https://vignesh2027.github.io/VORTEXRAG",
        "Source": "https://github.com/vignesh2027/VORTEXRAG",
        "Tracker": "https://github.com/vignesh2027/VORTEXRAG/issues",
    },
)
