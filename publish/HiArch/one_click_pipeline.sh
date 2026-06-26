#!/bin/bash

# sps matrix should end with .mtx
# .mtx and window.bed files are produced by the input-preparation scripts.

paraFile=$1
if [ -z "$paraFile" ] || [ ! -f "$paraFile" ]; then
    echo "Parameter file not found: $paraFile"
    echo "Usage: bash one_click_pipeline.sh /path/to/parameters.txt"
    exit 1
fi

source "$paraFile"
if [ -z "$basepath" ] || [ -z "$script_dir" ] || [ -z "$python_path" ] || [ -z "$torch_path" ]; then
    echo "Missing required settings in parameter file: basepath, script_dir, python_path, torch_path"
    exit 1
fi

log_tail_lines=${HIARCH_LOG_TAIL_LINES:-40}
processed_samples=0

run_step() {
    local step_label=$1
    local log_file=$2
    shift 2

    echo "  [RUN] ${step_label}"
    "$@" > "$log_file" 2>&1
    local status=$?
    if [ $status -ne 0 ]; then
        echo "  [ERROR] ${step_label} failed. Log: ${log_file}"
        echo "  Last ${log_tail_lines} log lines:"
        tail -n "$log_tail_lines" "$log_file"
        exit $status
    fi
    echo "  [OK] ${step_label}"
}

require_output() {
    local step_label=$1
    local output_file=$2
    if [ ! -s "$output_file" ]; then
        echo "  [ERROR] ${step_label} did not create expected output: ${output_file}"
        exit 1
    fi
}

for species in "$basepath"/*
do
    if [ ! -d "$species" ]; then
        continue
    fi
    if [ ! -d "$species/sps_mtx" ]; then
        continue
    fi

    samples=("$species"/sps_mtx/*_normalized.mtx)
    if [ ! -e "${samples[0]}" ]; then
        continue
    fi

    species_name=$(basename "$species")
    log_dir="$species/result/logs"
    mkdir -p "$log_dir"
    echo "[SPECIES] ${species_name}"

    for sample in "${samples[@]}"
    do
        processed_samples=$((processed_samples + 1))
        name=$(basename "$sample" .mtx)
        echo "[SAMPLE] ${name}"

        step="normDis"
        mkdir -p "$species/result/${step}/mtx" "$species/result/${step}/figure"
        run_step "$step" "$log_dir/${name}.${step}.log" \
            "$python_path" "$script_dir/NormDis.py" \
            -f "$species/sps_mtx/${name}.mtx" \
            -w "$species/sps_mtx/${name/_normalized/}.window.bed" \
            -o "$species/result/${step}/mtx/${name}" \
            -fo "$species/result/${step}/figure/${name}" \
            -df "$normdis_df" \
            -cmd "$normdis_cmd"
        require_output "$step" "$species/result/${step}/mtx/${name}.de_ode.mtx"

        pre_step="normDis"
        step="correctMap"
        mkdir -p "$species/result/${step}/mtx" "$species/result/${step}/figure"
        correctmap_cmd=(
            "$python_path" "$script_dir/CorrectMap.py"
            -f "$species/result/${pre_step}/mtx/${name}.de_ode.mtx"
            -w "$species/result/${pre_step}/mtx/${name}.window.bed"
            -o "$species/result/${step}/mtx/${name}"
            -fo "$species/result/${step}/figure/${name}"
            -drc "$correctmap_drc"
        )
        if [ -n "${correctmap_ac:-}" ]; then
            read -r -a correctmap_ac_args <<< "$correctmap_ac"
            correctmap_cmd+=( -ac "${correctmap_ac_args[@]}" )
        fi
        run_step "$step" "$log_dir/${name}.${step}.log" "${correctmap_cmd[@]}"
        require_output "$step" "$species/result/${step}/mtx/${name}.clean_de_ode.mtx"

        pre_step="correctMap"
        step="checkerBoard"
        mkdir -p "$species/result/${step}/mtx" "$species/result/${step}/figure"
        run_step "$step" "$log_dir/${name}.${step}.log" \
            "$python_path" "$script_dir/Checkerboard.py" \
            -f "$species/result/${pre_step}/mtx/${name}.clean_de_ode.mtx" \
            -w "$species/result/${pre_step}/mtx/${name}.clean_window.bed" \
            -o "$species/result/${step}/mtx/${name}.checkerBoard" \
            -sd "$checkboard_sd" \
            -fo "$species/result/${step}/figure/${name}"

        pre_step="correctMap"
        step="globalFolding_s1"
        mkdir -p "$species/result/${step}/mtx" "$species/result/${step}/figure"
        gf_s1_cmd=(
            "$python_path" "$script_dir/GF_S1_get_center.py"
            -f "$species/result/${pre_step}/mtx/${name}.clean_de_ode.mtx"
            -w "$species/result/${pre_step}/mtx/${name}.clean_window.bed"
            -o "$species/result/${step}/mtx/${name}"
            -am "$GF_S1_am"
            -ue "$GF_S1_ue"
            -fo "$species/result/${step}/figure/${name}"
        )
        if [ -n "${GF_S1_ac:-}" ]; then
            gf_s1_cmd+=( -ac "$GF_S1_ac" )
        fi
        run_step "$step" "$log_dir/${name}.${step}.log" "${gf_s1_cmd[@]}"
        require_output "$step" "$species/result/${step}/mtx/${name}.filt_ode.mtx"
        require_output "$step" "$species/result/${step}/mtx/${name}.anchors.txt"

        pre_step="globalFolding_s1"
        step="globalFolding_s2"
        mkdir -p "$species/result/${step}/mtx" "$species/result/${step}/figure"
        run_step "$step" "$log_dir/${name}.${step}.log" \
            "$torch_path" "$script_dir/GF_S2_get_score.py" \
            -f "$species/result/${pre_step}/mtx/${name}.filt_ode.mtx" \
            -w "$species/result/${pre_step}/mtx/${name}.window.bed" \
            -af "$species/result/${pre_step}/mtx/${name}.anchors.txt" \
            -o "$species/result/${step}/mtx/${name}" \
            -fo "$species/result/${step}/figure/${name}"
    done
done

if [ "$processed_samples" -eq 0 ]; then
    echo "[ERROR] No input samples found under: $basepath"
    echo "Expected files like: $basepath/<species>/sps_mtx/<sample>_normalized.mtx"
    echo "Expected matching windows: $basepath/<species>/sps_mtx/<sample>.window.bed"
    echo "If you only have balanced .mcool files, run scripts/prepare_hiarch_from_mcool.py first."
    exit 1
fi
