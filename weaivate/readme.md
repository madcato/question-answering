# Weavitae

Use this software to store and search embeddings from models like Bloom and OpenaAI. This embeddings are vectors with a hugh number of dimensions.

## Info
- [Weavitate on Github](https://github.com/semi-technologies/weaviate)
- [Weaviate home web](https://weaviate.io)
- [Getting started](https://weaviate.io/developers/weaviate/current/getting-started/index.html)
- [Installation](https://weaviate.io/developers/weaviate/current/installation/index.html)
- [Text/Image search for similar products](https://github.com/EsraaMadi/similarity-search-weaviate)
- [Python client documentation](https://weaviate-python-client.readthedocs.io/en/stable/#)
- [Using GPU from a docker container?](https://stackoverflow.com/questions/25185405/using-gpu-from-a-docker-container)

## Install

1. Install [`nvidia-container-toolkit` by following this guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
2. Download the last `docker-compose.yml` with `curl -o docker-compose.yml "https://configuration.semi.technology/v2/docker-compose/docker-compose.yml?modules=standalone&runtime=docker-compose&weaviate_version=v1.17.0"`
3. run `docker-compose up -d`


## Run

_(Optional: run .py files in the current directory to try it)_

### Run qa

1. `docker-compose up -d`
2. `python3 qa/load-db.py` (Only first time)
3. `python3 qa/main.py`