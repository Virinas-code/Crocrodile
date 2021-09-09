# Crocrodile
## Patches
### patch-001
Fix 0D array saving with numpy : save with builtin open().
```py
open(f"nns/{loop}-b5.csv", 'w').write(str(float(self.neural_networks[loop].b5)))  # patch-001
#####
self.neural_networks[loop].b5 = numpy.array(float(open(f"nns/{loop}-b5.csv").read()))  # patch-001
```
