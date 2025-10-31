#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
