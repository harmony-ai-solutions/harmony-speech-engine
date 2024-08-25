#!/bin/bash -e

echo 'Starting Harmony Speech Engine API server...'

CMD="python3 -m harmonyspeech.endpoints.openai.api_server
             --host 0.0.0.0
             --port 12080
             --download-dir ${HF_HOME:?}/hub
             ${CMD_ADDITIONAL_ARGUMENTS}"

# set umask to ensure group read / write at runtime
umask 002

set -x

exec $CMD
