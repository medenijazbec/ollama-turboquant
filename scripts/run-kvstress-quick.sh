#!/bin/sh
set -eu

PROFILE=quick exec "$(dirname "$0")/run-kvstress-ab.sh"
