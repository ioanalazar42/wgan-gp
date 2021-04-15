#!/bin/bash
# attaches to first session if no session name provided
if [ $# -eq 0 ]
  then
    echo "tmux list-sessions"
    SESH=$(tmux list-sessions)
    echo "${SESH}"
    WINDW="${SESH%%:*}"
    echo "connecting to first active session -> $WINDW"
    echo "tmux attach -t $WINDW"
    tmux attach -t $WINDW
  else
    echo "tmux attach -t $1"
    tmux attach -t $1
  fi
