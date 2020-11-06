#!/bin/bash

# Populate environment variables for .env
export $(egrep -v '^#' .env | xargs)

python3 train.py