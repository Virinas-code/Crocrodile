# Crocrodile
## Patches
### patch-001
Fix 0D array saving with numpy : save with builtin open().
```py
open(f"nns/{loop}-b5.csv", 'w').write(str(float(self.neural_networks[loop].b5)))  # patch-001
#####
self.neural_networks[loop].b5 = numpy.array(float(open(f"nns/{loop}-b5.csv").read()))  # patch-001
```
### patch-002 - Accept random color challenges
```py
# crocrodile/client/__init__.py, line 282
if event['challenge']['speed'] in SPEEDS and event['challenge']['variant']['key'] in VARIANTS and not event['challenge']['id'] in colors and event['challenge']['challenger']['id'] != "crocrodile" and event['challenge']['color'] != 'random':  # patch-002
# to
if event['challenge']['speed'] in SPEEDS and event['challenge']['variant']['key'] in VARIANTS and not event['challenge']['id'] in colors and event['challenge']['challenger']['id'] != "crocrodile":  # patch-002
```
