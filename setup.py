from setuptools import setup, find_packages

setup(
    name="godel0",
    version="0.4.0",
    description="Godel0: Self-improving coding agent with built-in Proposer and SWE-smith",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "openai>=1.0",
        "anthropic>=0.34",
        "backoff>=2.0",
        "GitPython>=3.1",
        "pathspec>=0.12",
        "rich>=13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
        ],
    },
    entry_points={
        "console_scripts": [
            "godel0=godel0.cli:main",
        ],
    },
)
