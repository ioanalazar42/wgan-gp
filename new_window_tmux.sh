#!/bin/bash
tmux new-window -t $1 -n $2
echo "Window $2 attached to session $1. Run \"tmux attach -t $1\""
