#!/usr/bin/env bash
set -Eeuo pipefail
dir="/projects/compures/alexandre/disdiff_adapters/disdiff_adapters/scripts/train"

loss="vae_nce"
version="md/md_with_val"
gpu="1"
session="md_train_$(date +%H%M%S)"
cmd="/projects/compures/alexandre/disdiff_adapters/disdiff_adapters/scripts/train/train_md.sh"

betas=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --loss)    loss="$2"; shift 2 ;;
    --version) version="$2"; shift 2 ;;
    --gpu)     gpu="$2";  shift 2 ;;
    --session) session="$2"; shift 2 ;;
    --) shift; break ;;
    -*) echo "Option inconnue: $1" >&2; exit 2 ;;
    *) betas+=("$1"); shift ;;
  esac
done

declare -a windows=()
for i in "${!betas[@]}"; do
  b="${betas[i]}"
  windows[i]="b_${b//./p}"
done

tmux new-session -d -s "$session" -n "${windows[0]}"
tmux send-keys -t "$session:${windows[0]}" "${cmd} ${betas[0]} ${loss} ${version} ${gpu}" C-m 

for i in "${!windows[@]}"; do
  ((i == 0)) && continue 
  tmux new-window -d -t "$session" -n "${windows[i]}"
  tmux send-keys -t "$session:${windows[i]}" "${cmd} ${betas[i]} ${loss} ${version} ${gpu}" C-m 
  
done

echo "tmux a -t ${session}"