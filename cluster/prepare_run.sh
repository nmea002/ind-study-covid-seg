#!/bin/bash
# Unzip a dataset zip into its own run folder and patch the copy for the cluster environment.
# Run on the login node, once per dataset:
#   bash prepare_run.sh qata   ../LViT_QaTa-COV19-v2.zip
#   bash prepare_run.sh mosmed ../LViT_MosMedData.zip
# Result: <project>/runs/<name>/LViT/  (training output lands in there too)
set -euo pipefail

NAME="$1"
ZIP="$(realpath "$2")"
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="$(dirname "$HERE")/runs/$NAME"  # inside the project folder

if [ -d "$RUN/LViT" ]; then
    echo "$RUN/LViT already exists -- delete it first if you really want a fresh unzip (it may hold trained models)."
    exit 1
fi
mkdir -p "$RUN"
python3 -m zipfile -e "$ZIP" "$RUN"  # the login node has no `unzip`
cd "$RUN/LViT"

# mxnet-free replacement for the bert-embedding package (see bert_embedding.py)
cp "$HERE/bert_embedding.py" .
# ...and embed all descriptions once at dataset creation, instead of per sample inside the DataLoader workers
sed -i 's/^\(\s*\)self.bert_embedding = BertEmbedding()$/&\n\1self.bert_embedding.precompute(row_text.values())/' Load_Dataset.py
# scipy.ndimage.interpolation is a deprecated alias; zoom lives in scipy.ndimage
sed -i 's/from scipy.ndimage.interpolation import zoom/from scipy.ndimage import zoom/' Load_Dataset.py
# torch >= 2.6 defaults to weights_only=True, which rejects the numpy values saved in the checkpoint
sed -i "s/torch.load(\(.*\), map_location='cuda')/torch.load(\1, map_location='cuda', weights_only=False)/" test_model.py train_model.py
# write history.csv (per-epoch train/val loss and Dice) for the training curves
python3 "$HERE/patch_lvit.py"
# log every 50 batches instead of every batch
sed -i 's/^print_frequency = .*/print_frequency = 50/' Config.py

TASK=$(grep "^task_name" Config.py | sed "s/.*= *'\(.*\)'.*/\1/")
echo "prepared $RUN/LViT  (task_name = $TASK)"
for f in Train_Folder Val_Folder Test_Folder; do
    echo "  $f: $(ls "datasets/$TASK/$f/img" | wc -l) images"
done
