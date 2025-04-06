from setuptools import setup, find_packages

setup(
    name="spamfilter",
    version="0.1.0",
    description="Spam classification library with TF-IDF and Transformer support",
    author="tizhproger",
    author_email="tizhproger.development@gmail.com",
    url="https://github.com/tizhproger/spamfilter",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "transformers>=4.37.0",
        "datasets",
        "torch",
        "joblib",
        "tqdm",
        "nltk"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
