#!/bin/sh
set -eu

PROFILE=impact exec "$(dirname "$0")/run-kvstress-ab.sh"
