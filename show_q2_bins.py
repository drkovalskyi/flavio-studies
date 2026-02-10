import flavio
bins = {}
for m in flavio.Measurement.instances.values():
    for k in m.all_parameters:
        if isinstance(k, tuple) and len(k)>=3 and k[0].endswith('(Bs->phimumu)'):
            bins.setdefault(k[0], set()).add((float(k[1]), float(k[2])))
for obs in sorted(bins.keys()):
    print(obs)
    for b in sorted(bins[obs]):
        print(" ", b)
    print()
