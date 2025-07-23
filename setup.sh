#!/bin/bash

# Debian e Ubuntu
sudo apt-get update
sudo apt-get install libportaudio2 portaudio19-dev libportaudiocpp0 -y

pip install -r requirements.txt --break-system-packages