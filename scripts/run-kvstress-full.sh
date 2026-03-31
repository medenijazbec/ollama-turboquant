#!/bin/sh
set -eu

PROFILE=full exec "$(dirname "$0")/run-kvstress-ab.sh"
