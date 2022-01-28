#!/bin/sh

ROOT_DIR='/root_dir'
echo "ROOT_DIR : ${ROOT_DIR}"

uvicorn --reload --reload-dir "${ROOT_DIR}/data/interim" --reload-include '*_dataset' app.fast:app --host 0.0.0.0
