from setuptools import setup, find_packages

setup(
    name="clotho_env",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "ipykernel",
        "pandas",
        "plotly",
        "datetime",
        "nbformat",
        "sshtunnel",
        "psycopg2-binary",
        "gunicorn",
        "langdetect",
        "matplotlib",
        "networkx",
        "pyvis",
    ],
    extras_require={"dev": ["pylint", "black"]},
)