#!/bin/bash
prpl=$'\e[1;31m'
end=$'\e[0m'
printf "Create new tmux session:\n${prpl}./create_tmux_session.sh <session_name> <window1> <window2> ...${end}\n\n"
printf "Add one new window to existing session:\n${prpl}./new_window_tmux.sh <session_name> <window_name>${end}\n\n"
printf "Attach to first existing tmux session:\n${prpl}./attach_tmux.sh${end}\n\n"
printf "Attach to specified tmux session:\n${prpl}./attach_tmux.sh <session_name>${end}\n"
