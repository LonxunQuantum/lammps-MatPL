#!/bin/bash

# This script is used to check the precision of the NEP GPU sources
#     - single: use single precision for NEP GPU
#     - double: use double precision for NEP GPU

set -euo pipefail

SYMBOLS=$(nm -C lmp | grep "update_potential" || true)
echo "$SYMBOLS"
echo
echo "Please check the NEPKK symbol line:"
echo "  NEPKK::update_potential(float*, ...)  -> single precision"
echo "  NEPKK::update_potential(double*, ...) -> double precision"


