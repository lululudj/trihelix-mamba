import sys; sys.stdout.reconfigure(encoding="utf-8")
with open("models/common.py","r",encoding="utf-8") as f:
    c = f.read()

old = "preds = []\n        h = h_t\n        for _ in range(self.n_future):\n            h = self.predictor(h)\n            preds.append(self.proj_out(h))"

new = "preds = []\n        h = torch.zeros(h_t.shape[0], self.predictor.hidden_size, device=h_t.device)\n        x = h_t\n        for _ in range(self.n_future):\n            h = self.predictor(x, h)\n            preds.append(self.proj_out(h))\n            x = preds[-1]"

if old in c:
    c = c.replace(old, new)
    print("Fixed JEPA forward")
else:
    print("Pattern not found")

with open("models/common.py","w",encoding="utf-8") as f:
    f.write(c)
