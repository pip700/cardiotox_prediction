#!/bin/bash
apt-get update
apt-get install -y libxrender1 libsm6 libxext6
pip install --upgrade pip && pip install -r requirements.txt
