# Bash >= 4 (associative arrays)
# Usage:
#   parse_train_tree                              # uses default logs root, writes ./train.log
#   parse_train_tree /path/to/logs                # writes ./train.log
#   parse_train_tree /path/to/logs /tmp/train.log

parse_train_tree() {
  local logs_root="${1:-/projects/compures/alexandre/disdiff_adapters/disdiff_adapters/logs}"
  local out_file="${2:-train.log}"
  logs_root="${logs_root%/}"

  : > "$out_file" || { echo "Cannot write to $out_file" >&2; return 1; }

  declare -A max_epoch

  local epdir base ep train_dir rel beta_t_part beta_t dataset dim_seg dim_s run_folder beta_s beta_s_clean key

  while IFS= read -r -d '' epdir; do
    base="$(basename "$epdir")"
    ep="${base#epoch_}"
    [[ "$ep" =~ ^[0-9]+$ ]] || continue

    train_dir="$(dirname "$epdir")"
    rel="${train_dir#"$logs_root"/}"

    # beta_t from: x_with_beta_t100
    beta_t_part="${rel%%/*}"
    beta_t="$(printf '%s' "$beta_t_part" | sed -n 's/^x_with_beta_t\([0-9]\+\).*$/\1/p')"
    [[ -n "$beta_t" ]] || continue

    # dataset = 2nd path component
    dataset="$(printf '%s' "$rel" | cut -d'/' -f2)"
    case "$dataset" in
      celeba|mpi3d|cars3d|shapes|dsprites) ;;
      *) continue ;;
    esac

    # dim_s from: test_dim_s126
    dim_seg="$(printf '%s\n' "$rel" | awk -F/ '{for(i=1;i<=NF;i++){if($i ~ /^test_dim_s/){print $i; exit}}}')"
    dim_s="$(printf '%s' "$dim_seg" | sed -n 's/^test_dim_s\([0-9]\+\).*$/\1/p')"
    [[ -n "$dim_s" ]] || continue

    # beta_s = first float after beta=( ... ) inside training folder name
    run_folder="$(basename "$train_dir")"
    beta_s="$(printf '%s\n' "$run_folder" | sed -n 's/.*beta=(\([^,)]*\).*/\1/p')"
    [[ -n "$beta_s" ]] || continue
    # clean "100.0" -> "100" (keeps non-trivial decimals like 0.001)
    beta_s_clean="$(printf '%s' "$beta_s" | sed 's/\.0*$//')"

    key="${dataset}|${dim_s}|${beta_t}|${beta_s_clean}"

    if [[ -z "${max_epoch[$key]+x}" || "$ep" -gt "${max_epoch[$key]}" ]]; then
      max_epoch["$key"]="$ep"
    fi
  done < <(
    find "$logs_root" -type d \
      -path "*/loss_vae_nce/factor_s=-1/batch32/test_dim_s*/x_epoch=*beta=*/epoch_*" \
      -name 'epoch_*' -print0 2>/dev/null
  )

  {
    for key in "${!max_epoch[@]}"; do
      IFS='|' read -r dataset dim_s beta_t beta_s_clean <<< "$key"
      printf "%s, dims%s, beta_t %s, beta_s %s  epoch %s\n" \
        "$dataset" "$dim_s" "$beta_t" "$beta_s_clean" "${max_epoch[$key]}"
    done
  } | sort > "$out_file"

  echo "Wrote $(wc -l < "$out_file") lines to $out_file"
}

