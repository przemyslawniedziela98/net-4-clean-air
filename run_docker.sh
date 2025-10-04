set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_DIR="${PROJECT_DIR}/qdrant_storage"
mkdir -p "${STORAGE_DIR}"

docker network inspect clean_air_net >/dev/null 2>&1 || \
    docker network create clean_air_net

if [ "$(docker ps -aq -f name=qdrant_clean_air)" ]; then
    docker stop qdrant_clean_air >/dev/null 2>&1 || true
    docker rm qdrant_clean_air >/dev/null 2>&1 || true
fi

docker run -d \
  --name qdrant_clean_air \
  --network clean_air_net \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "${STORAGE_DIR}:/qdrant/storage" \
  qdrant/qdrant:latest
