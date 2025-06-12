#!/bin/bash

# Debian e Ubuntu
sudo apt-get update
sudo apt-get install libportaudio2 portaudio19-dev libportaudiocpp0 -y

## RedHat e Fedora
sudo dnf install portaudio-devel