#!/bin/bash

if [ ! -d "log" ]; then
  mkdir log
fi
python3  train_T5.py

