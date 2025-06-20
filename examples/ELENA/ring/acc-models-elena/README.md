# ELENA Ring Optics Repository
>
> (Known) contributors:
>   - Pavel Belochitskii (since 1998(?))
>   - Olav Ejner Berrig 
>   - Laurette Ponce (since 2018) 
>   - Davide Gamba (since 2019)
>

===

The initial version of this [repository](https://gitlab.cern.ch/acc-models/acc-models-elena) is what was in AFS (/afs/cern.ch/eng/ps/cps/ELENA/2016/) and SVN (https://svn.cern.ch/reps/camif/trunk/ELENA).
The present model should be pretty compatible with Pavel's model used for the design and construction of the ring (See ELENA Design Report - [CERN-2014-002](https://cds.cern.ch/record/1694484/files/CERN-2014-002.pdf))

## Repository Description

The MAD-X element definitions and sequences are contained in the single `elena.seq` file, while `elena.dbx` (will) contains the apertures of the different type of elements.

The typical machine configurations, i.e. quadrupoles strengths and relative optics and scripts to generate them, are stored in the `scenarios` folder.

Additionally:
- `survey` folder contains MAD-X script (`Survey_preparation.madx`) to generated the survey files, as well as tables of element positions in TFS format. Note that:
    - the element s-position in the survey tables is relative to the end of the element
    - the element lengths are the magnetic lengths
    - initial point is the start of section 1, i.e. the exit of LNR.MBHEK.0640 (i.e. the dipole of section 6)
    - output survey tables are produced in the global CERN coordinate system (`ELENA_input_for_GEODE.sur`), in a local coordinate system, i.e. starting from 0,0,0,0,0,0 (`ELENA_0.sur`).
- `operation` folder contains the JMAD XML configuration. 
- `tools` folder containing a few MAD-X scripts to split parts of the sequence and/or for other studies.
- `madxTesting` contains an example (and its output) for MAD-X testing purposes. **It is not meant to be changed/used!** 
- `_scripts` folder contains a few python scripts used to build the ACC-Models-ELENA website and the JMAD XML configuration.
- `.gitlab-ci.yml` file contains the scripts which are run by gitlab when you push a new version of the model. Typically this is needed to re-populate the acc-models website.
- `Makefile` is a simple **MakeFile** to easy the building process of models/outputs

## Generic GIT Commands:

The repository is managed with git, enabling version-control in an easy-to access central repository.
A summary of some basic git commands are given below.
If you choose, you can also use a graphical client or directly download files from the repository web-page.

- download the project 
```
git clone https://gitlab.cern.ch/acc-models/acc-models-elena.git
```

- update the local repository with the remote repository
```
git pull
```

- add a file
```
git add newFile.txt
```

- commit the changes to the LOCAL repository
```
git commit -a -m "my comment"
```
note that the -a option means "all", i.e. it commits all files that where added (see procedure above) and the ones that were already in the repository but you modified.

- synchronize the LOCAL repository to the REMOTE repository
```
git push
```
