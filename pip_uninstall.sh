#/bin/sh

PYTHON_PATH=$(which python)
BASE_DIR=${PYTHON_PATH%%/bin/*}
echo $BASE_DIR

rm -r build
rm -r src/leapct.egg-info/
rm -r ${BASE_DIR}/lib/python3.*/site-packages/leap*

