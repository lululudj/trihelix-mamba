import sys; sys.stdout.reconfigure(encoding="utf-8")

with open("models/three_chain.py", "r", encoding="utf-8") as f:
    t = f.read()

# Add JEPA import
old_import = "from .common import ("
new_import = "from .common import (\n    CTM_Oscillator, JEPA_Predictor,"
t = t.replace(old_import, new_import)

# Add JEPA init after time_head
old_time = "self.time_head = TimeAuxHead(d_model, max_T)"
new_time = "self.time_head = TimeAuxHead(d_model, max_T)\n        self.jepa = JEPA_Predictor(d_state_total=d_model, n_future=3, hidden=64)"
t = t.replace(old_time, new_time)

# Add JEPA prediction after bind
old_bind = "h_bind = self.bind(h_s, h_c, h_t)"
new_bind = "h_bind = self.bind(h_s, h_c, h_t)\n        h_last = h_bind[:, -1]\n        jepa_pred = self.jepa(h_last)"
t = t.replace(old_bind, new_bind)

# Add jepa_pred to aux dict
old_aux = '"bp_info": bp_info}'
new_aux = '"bp_info": bp_info, "jepa_pred": jepa_pred, "h_bind_last": h_bind[:, -3:]}'
t = t.replace(old_aux, new_aux)

# Add JEPA loss before time loss
old_time_loss = 'if "time_pred" in aux and aux["time_pred"] is not None:'
new_time_loss = '''if "jepa_pred" in aux and "h_bind_last" in aux:
            jepa_loss = self.jepa.compute_loss(aux["jepa_pred"], aux["h_bind_last"])
            total = total + 0.3 * jepa_loss
            info["jepa_loss"] = jepa_loss.item()
        
        if "time_pred" in aux and aux["time_pred"] is not None:'''
t = t.replace(old_time_loss, new_time_loss)

with open("models/three_chain.py", "w", encoding="utf-8") as f:
    f.write(t)

print("JEPA integrated into ThreeChain")
