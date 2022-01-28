#!/bin/sh
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
echo $timestamp
cd /root/${WORKING_DIR}
docker-compose down
docker image prune -f
git pull
docker-compose build
docker-compose up -d