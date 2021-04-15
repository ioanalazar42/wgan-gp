#!/bin/bash
tmux new-session -s $1 -n $2 -d
for var in "${@:3}"
do
  tmux new-window -t $1 -n $var
done
echo "To connect to session: tmux attach -t $1"
