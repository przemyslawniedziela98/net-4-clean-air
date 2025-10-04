set -e
VENV_NAME="cleanair"

if [ ! -d "${VENV_NAME}" ]; then
    python3 -m venv "${VENV_NAME}"
fi

source "${VENV_NAME}/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

export FLASK_APP=run.py
export FLASK_ENV=development
export PYTHONPATH=.

flask run --host=0.0.0.0 --port=5001
