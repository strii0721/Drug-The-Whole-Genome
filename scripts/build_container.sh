#!/bin/bash
docker run --gpus all -v $(pwd):/project --shm-size=32g -it dtwg:latest bash