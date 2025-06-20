# Simple MakeFile to build different parts of this repository
#
# For some python scripts, make sure you installed the packages in `_scripts/packages.txt`
#  or use the provided `make pypackage` target
# 
# Davide - May 2021

# MAD-X executable:
MADX = ~/madx/madx-macosx64-gnu
#MADX = ~/madx/madx-macosx64-intel

all : optics survey bokeh xml
 
optics : elena.seq elena.dbx scenarios/*/*.madx scenarios/*/*.str scenarios/*/*.beam
	for i in $$(ls scenarios); do \
	   ls scenarios/$$i ; \
	   cd scenarios/$$i ; \
	   $(MADX) < $$i.madx ; \
	   cd ../../ ; \
	done \

survey : elena.seq survey/Survey_preparation.madx
	cd survey; $(MADX) < Survey_preparation.madx

madxtest : elena.seq elena.dbx madxTesting/test.madx madxTesting/test.str
	cd madxTesting ; $(MADX) < test.madx ; cd .. ;

xml :
	python _scripts/create_XML.py

bokeh :
	python _scripts/create_bokeh.py

pypackage :
	for package in `cat _scripts/packages.txt`; do pip install $$package; done
