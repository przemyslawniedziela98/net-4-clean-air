set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STORAGE_DIR="${PROJECT_DIR}/qdrant_storage"
mkdir -p "${STORAGE_DIR}"

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed. Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running. Please start Docker Desktop or the Docker daemon."
    exit 1
fi

docker-compose -f "${PROJECT_DIR}/docker-compose.yml" up -d --build