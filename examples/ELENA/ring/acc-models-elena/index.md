---
template: overrides/main.html
---

ELENA Ring Optics
===

This [repository](https://gitlab.cern.ch/acc-models/acc-models-elena) contains the model of ELENA ring build on the basis of Pavel's model, coming from his private AFS folder.
The initial version comes from AFS (/afs/cern.ch/eng/ps/cps/ELENA/2016/) and SVN (https://svn.cern.ch/reps/camif/trunk/ELENA), from which the style was adapted to merge present AD/ELENA Standards

ELENA is a small hexagonal ring. The lattice is made of a 4 simple structures, each made of one doublet and one singlet quadrupoles. The other two straight sections host the injection and the electron cooler.

ELENA decelerates antiprotons from 100 MeV/c to 13.7 MeV/c. The magnetic cycle is therefore composed of:

- injection at 100 MeV/c
- deceleration and e-cooling at 35 MeV/c
- deceleration and e-cooling at 13.7 MeV/c + extraction

> For more details on ELENA design, see *Chohan, V (ed.) et al.* Extra Low ENergy Antiproton (ELENA) ring and its Transfer Lines: Design Report - [CERN-2014-002](https://cds.cern.ch/record/1694484/) - 2014

##  Optics

> WARNING!!! OPTICS MODEL IS BEING UPDATED - September 2021

ELENA has **single optic** used all along the cycle. Presently, in the optics repository a single optics is provided assuming the injection energy (i.e. p=0.1 GeV/c; beta_rel=0.1059786).

> Note that in the plot below, dispersion is computed with DX from MAD-X * beta_rel.

TFS table of the optics available [here](scenarios/highenergy/highenergy.tfs){target=_blank}.
<object width="100%" height="550" data="scenarios/highenergy/highenergy.html"></object> 


## MAD-X Example on SWAN

You can directly open a MAD-X example in [SWAN](https://cern.ch/swanserver/cgi-bin/go?projurl=file://eos/project/a/acc-models/public/elena/MADX_example_elena.ipynb){target=_blank}.


