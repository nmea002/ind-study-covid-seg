"""Patch an unzipped LViT run folder so training also writes history.csv (one row per epoch:
epoch, train_loss, train_dice, val_loss, val_dice) into the session folder, for run_report.py's training curves.
Called by prepare_run.sh from inside the run's LViT folder. Each patch must match exactly once.
"""
import sys

PATCHES = [
    ("train_model.py",
     "        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)  # sup\n",
     "        train_loss, train_dice = train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None,\n"
     "                                                 model_type, logger)  # sup\n"),
    ("train_model.py",
     "            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,\n"
     "                                                 optimizer, writer, epoch, lr_scheduler, model_type, logger)\n",
     "            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,\n"
     "                                                 optimizer, writer, epoch, lr_scheduler, model_type, logger)\n"
     "        with open(config.save_path + 'history.csv', 'a') as f:\n"
     "            if epoch == 0:\n"
     "                f.write('epoch,train_loss,train_dice,val_loss,val_dice\\n')\n"
     "            f.write('{},{},{},{},{}\\n'.format(epoch + 1, float(train_loss), float(train_dice),\n"
     "                                               float(val_loss), float(val_dice)))\n"),
]

for path, old, new in PATCHES:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if new in src:
        continue  # already patched
    if src.count(old) != 1:
        sys.exit(f"patch target not found exactly once in {path}:\n{old}")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(src.replace(old, new))
print("patched train_model.py: writes history.csv per epoch")
