# Model artifacts

The checkpoint `efficientnet_asl_weights.pth` is present and can run without `labels.json`.

To show sign names instead of `class_0` … `class_28`, optionally provide `labels.json` in the original training output order. It must be either a JSON array:

```json
["A", "B", "C"]
```

or an object containing that array under `labels`:

```json
{"labels": ["A", "B", "C"]}
```

Set `LABELS_PATH=model/labels.json` in `.env` after adding it. The number and ordering of labels must exactly match the checkpoint's 29 classifier outputs. The checkpoint alone cannot reveal that mapping.
