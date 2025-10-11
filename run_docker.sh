set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STORAGE_DIR="${PROJECT_DIR}/qdrant_storage"
mkdir -p "${STORAGE_DIR}"

docker-compose -f "${PROJECT_DIR}/docker-compose.yml" up -d --build
