#!/bin/sh
set -eu

PROFILE=staircase \
KVMODES="${KV_MODES:-f16,q8_0,q4_0,tq25,tq35}" \
exec "$(dirname "$0")/run-kvstress-ab.sh"
