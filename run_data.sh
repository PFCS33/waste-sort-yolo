#!/bin/bash
# Concise data processing wrapper

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

case "${1:-help}" in
    setup)
        [ -f scripts/data/requirements.txt ] && pip install -r scripts/data/requirements.txt || echo "No requirements.txt found"
        ;;
    download|transform|merge|all)
        python -m scripts.data "$@"
        ;;
    hierarchy)
        python -m scripts.data hierarchy "${@:2}"
        ;;
    test-*)
        func=${1#test-}  # Remove 'test-' prefix
        python -m scripts.data test --func "$func" "${@:2}"
        ;;
    help|--help|-h|*)
        echo "Usage: $0 [command] [options]"
        echo "Commands: setup, download, transform, merge, all, hierarchy, test-{draw,count,distribution}, help"
        echo "Examples:"
        echo "  $0 all"
        echo "  $0 hierarchy --config ./scripts/data/hierarchy/config.yaml"
        echo "  $0 test-draw --image-path img.jpg --label-path label.txt"
        ;;
esac