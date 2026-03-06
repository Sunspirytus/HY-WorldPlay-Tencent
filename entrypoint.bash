#!/bin/bash


source /app/.bashrc
pwd
micromamba run --name base python gradio_app.py
