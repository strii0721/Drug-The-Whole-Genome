#!/usr/bin/env bash
# retrieval_all.sh — batch virtual screening for all Drug-The-Whole-Genome targets
# ============================================================================
# Run inside the dtwg Docker container (GPU required):
#     bash /project/retrieval_all.sh
# ============================================================================
set -euo pipefail

PROJ=/project
DATA="$PROJ/data/targets"
OUT="$PROJ/output/dti/retrieval_all"
DICT="$PROJ/dict"
RUNLOG="$OUT/run.txt"

# ---- embedded alias lookup (28 targets) -----------------------------------
declare -A ALIASES
ALIASES[5HT2A]="5-羟色胺受体2A (HTR2A, 5-HT2A)"
ALIASES[ADRB2]="β2肾上腺素受体 (ADRB2R, B2AR, BAR)"
ALIASES[ADRB3]="β3肾上腺素受体 (B3AR)"
ALIASES[ALDH1]="乙醛脱氢酶1 (ALDH1A1, ALDC, PUMB1)"
ALIASES[BTK]="布鲁顿酪氨酸激酶 (ATK, BPK, XLA, AGMX1)"
ALIASES[CB2]="大麻素受体2 (CNR2, CB-2)"
ALIASES[D2R]="多巴胺D2受体 (DRD2, D2DR)"
ALIASES[ESR2]="雌激素受体β (ER-BETA, ESRB, NR3A2)"
ALIASES[FEN1]="瓣状内切核酸酶1 (MF1, RAD2)"
ALIASES[GABAA_A1B2C2]="GABAA受体 α1β2γ2亚型"
ALIASES[GBA]="葡萄糖脑苷脂酶 (GCB, GLUC)"
ALIASES[GPR35]="G蛋白偶联受体35 (GPR35)"
ALIASES[GluN1_2A]="NMDA受体 GluN1/GluN2A亚型 (GRIN1/GRIN2A)"
ALIASES[GluN1_2B]="NMDA受体 GluN1/GluN2B亚型 (GRIN1/GRIN2B)"
ALIASES[IDH1]="异柠檬酸脱氢酶1 (IDP, IDCD, IDH, PICD)"
ALIASES[KAT2A]="组蛋白乙酰转移酶 KAT2A (GCN5, GCN5L2)"
ALIASES[MAPK1]="丝裂原活化蛋白激酶1 / ERK2 (PRKM2, p42-MAPK)"
ALIASES[MAPK10]="丝裂原活化蛋白激酶10 / JNK3 (SAPK1b, PRKM10)"
ALIASES[MTORC1]="mTOR复合体1 (MTOR + RPTOR + MLST8)"
ALIASES[NET]="去甲肾上腺素转运体 (SLC6A2, NAT1)"
ALIASES[OPRK1]="κ-阿片受体 (OPRK, KOR, KOR-1)"
ALIASES[P2X3]="P2X嘌呤受体3 (P2RX3)"
ALIASES[SCN9A]="电压门控钠通道 Nav1.7 (PN1, NENA)"
ALIASES[TRPM8]="冷/薄荷醇受体 TRPM8 (TRP-P8, LTRPC6)"
ALIASES[TRPV1]="辣椒素受体 (VR1, TRPV1)"
ALIASES[VDR]="维生素D受体 (NR1I1)"
ALIASES[XIAP]="X连锁凋亡抑制蛋白 (API3, BIRC4, hILP, IAP3)"
ALIASES[XOR]="黄嘌呤氧化还原酶 (XDH, XO)"

# ---- helper: seconds → h/m/s ---------------------------------------------
fmt_duration() {
    local s=$1
    printf "%dh %dm %ds" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

# ---- main -----------------------------------------------------------------
mkdir -p "$OUT"
START_TOTAL=$(date +%s)
START_TS=$(date '+%Y-%m-%d %H:%M:%S')

echo "Output directory: $OUT"
echo ""

# accumulate per-target timing lines
TIMING_LINES=""
TOTAL=0; SKIP=0; OK=0; FAIL=0

for tdir in "$DATA"/*/; do
    target=$(basename "$tdir")
    pocket="$tdir/PDB/pocket.lmdb"
    [ -f "$pocket" ] || continue
    TOTAL=$((TOTAL + 1))

    tout="$OUT/$target"
    res_file="$tout/$target.txt"
    prop_file="$tout/property.txt"

    if [ -f "$res_file" ]; then
        echo "[$TOTAL] $target — SKIP (already done)"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo "[$TOTAL] $target — screening..."

    T0=$(date +%s)
    mkdir -p "$tout"

    # --- retrieval ----------------------------------------------------------
    if CUDA_VISIBLE_DEVICES=0 python -u "$PROJ/unimol/retrieval.py" \
        --user-dir "$PROJ/unimol" \
        "$DICT" \
        --valid-subset test \
        --num-workers 8 --ddp-backend=c10d --batch-size 4 \
        --task drugclip --loss in_batch_softmax --arch drugclip \
        --max-pocket-atoms 511 \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
        --log-interval 100 --log-format simple \
        --mol-path mols.lmdb \
        --pocket-path "$pocket" \
        --fold-version 6_folds \
        --use-cache True \
        --save-path "$res_file"; then

        T1=$(date +%s)
        ELAPSED=$((T1 - T0))
        n_lines=$(wc -l < "$res_file")
        echo "       -> $res_file  (${n_lines} lines)  [$(fmt_duration $ELAPSED)]"
        OK=$((OK + 1))
        TIMING_LINES+=$(printf "  %-18s %s\n" "$target" "$(fmt_duration $ELAPSED)")
    else
        T1=$(date +%s)
        ELAPSED=$((T1 - T0))
        echo "       -> FAILED (exit code $?)  [$(fmt_duration $ELAPSED)]"
        FAIL=$((FAIL + 1))
        TIMING_LINES+=$(printf "  %-18s FAILED after %s\n" "$target" "$(fmt_duration $ELAPSED)")
        continue
    fi

    # --- property.txt -------------------------------------------------------
    alias_str="${ALIASES[$target]:-unknown}"
    cat > "$prop_file" <<-PROPEOF
Gene:     $target
Aliases:  $alias_str

Top-100 Molecules (by DrugCLIP score)
----------------------------------------
$(head -100 "$res_file" | awk -F',' '{
    printf "%4s  %-16s  %8.4f  %s\n", NR, $1, $3, $2
}')
PROPEOF
    echo "       -> $prop_file"

done

# ---- run.txt --------------------------------------------------------------
END_TOTAL=$(date +%s)
END_TS=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_ELAPSED=$((END_TOTAL - START_TOTAL))

cat > "$RUNLOG" <<-EOF
retrieval_all  run log
=====================
Started:   $START_TS
Finished:  $END_TS
Total:     $(fmt_duration $TOTAL_ELAPSED)

Summary:   $OK succeeded  /  $SKIP skipped  /  $FAIL failed  (out of $TOTAL)

Per-target:
$TIMING_LINES

Data Sources
============
Small-molecule library:  ChemDiv (1,648,137 molecules)
Target protein data:     ChEMBL (activity labels) + PDB (3D structures
                          used to encode pocket representations)
Result cutoff:           top 2% molecules per target (DrugCLIP score)
EOF

echo
echo "========== DONE =========="
echo "Total: $TOTAL  |  OK: $OK  |  Skipped: $SKIP  |  Failed: $FAIL"
echo "Total time: $(fmt_duration $TOTAL_ELAPSED)"
echo "Log:   $RUNLOG"
echo "Data:  $OUT"
