#!/bin/sh
set -eu

PROFILE=capacity \
KVMODES="${KV_MODES:-f16,tq25,tq35}" \
exec "$(dirname "$0")/run-kvstress-ab.sh"
