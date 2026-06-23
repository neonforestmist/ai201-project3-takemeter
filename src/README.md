# Source

Optional local source can go here.

The main training flow is expected to run in the copied Colab notebook. The
Milestone 6 sample-confidence export lives in
`scripts/export_milestone6_samples.py` because it is a reproducible project
utility rather than application source code.

`takemeter_app.py` is the optional stretch interface. It trains a lightweight
local classifier from the committed training split and serves a browser UI:

```bash
python3 src/takemeter_app.py --port 8765
```
