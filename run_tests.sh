set -e

VENV_NAME="cleanair"

if [ ! -d "${VENV_NAME}" ]; then
    python3 -m venv "${VENV_NAME}"
fi

source "${VENV_NAME}/bin/activate"

pip install -r requirements.txt
coverage run -m pytest -v tests/
coverage report -m
coverage html