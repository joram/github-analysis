#!/bin/bash

# rif python3-venv is not installed, install it
if ! command -v python3-venv &> /dev/null; then
    echo "python3-venv could not be found"
    sudo apt-get install python3-venv
    if [ $? -ne 0 ]; then
        echo "Failed to install python3-venv"
        exit 1
    fi
fi

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# if the .env doesn't exist, create it, empty it
if [ ! -f .env ]; then
    echo "" > .env
fi
#if it doesn't contain GITHUB_TOKEN, add it
if ! grep -q "GITHUB_TOKEN" .env; then
    echo "Open the following URL in your browser and copy the token:"
    echo "https://github.com/settings/personal-access-tokens/new?contents=read&metadata=read&pull_requests=read"
    read -p "Enter your GitHub token: " GITHUB_TOKEN
    echo "Adding GITHUB_TOKEN to .env"
    echo "GITHUB_TOKEN=$GITHUB_TOKEN" >> .env
fi


./analyis.py