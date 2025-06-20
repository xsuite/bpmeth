# Generic script to create operational scenarios xml files
#
# It is based on the following assumptions:
#  - repository has "scenarios" folder with sub-folders being all optics names
#  - in each scenario there is a single strength file called "<scenarioName>.str"
#  - repository has "operation" folder where output will be generated
#  - I guess MAD-X files should not call each other using some relative paths etc...
#
# It is meant to be run from the root of the repository. 
#
from yattag import Doc, indent
import os

# Simple function to avoid looking for hidden folders
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

#################################
# input parameters
#################################
# Name of the machine
machineName   = 'ELENA'

# Scripts to call to initialize model
commonInitialisationScripts = ['elena.seq', 'elena.dbx', 
    'tools/splitEle_installBPM.madx']

# Name of the default optics folder under 'scenarios' folder
default_optic       = 'highenergy'
# Sequence 
default_sequence    = 'ELENA'
# Range range start/end. If empty (or len() != 2), the whole sequence will be used.
default_range       = []
# Twiss options:
# - Initial beta functions. If empty, assuming closed optics, otherwise name of BETA0 variable
twiss_init_beta     = []
# - Calculate lattice functions at center of element:
twiss_center        = 'true'
# - Calculate chromatic functions
twiss_chrom         = 'true'

#################################

#################################
# Nothing to touch below here, in principle


# automatically generated file name according to machine name
outputFilename  = 'operation/'+machineName.lower()+'.jmd.xml'
allScenarios = listdir_nohidden('scenarios')

# Build XML file
doc, tag, text = Doc().tagtext()
with tag('jmad-model-definition', name = machineName):

    # define different optics according to different scenarios
    with tag('optics'):
        for scenario_ in allScenarios:
            with tag('optic', name = scenario_, overlay = 'false'):
                with tag('init-files'):
                    doc.stag('call-file', path = 'scenarios/'+scenario_+'/'+scenario_+'.str', parse='STRENGTHS')
                    doc.stag('call-file', path = 'scenarios/'+scenario_+'/'+scenario_+'.beam')

    # specify the default one
    doc.stag('default-optic ref-name="' + default_optic + '"')

    # define the sequence
    with tag('sequences'):
        with tag('sequence', name=default_sequence):
            with tag('ranges'):
                with tag('range', name='defaultRange'):
                    if len(default_range) == 2:
                        doc.stag('madx-range', first=default_range[0], last=default_range[1])
                    with tag('twiss-initial-conditions', name='default-twiss'):
                        doc.stag('chrom', value=twiss_chrom)
                        doc.stag('centre', value=twiss_center)
                        if len(twiss_init_beta) > 0:
                            doc.stag('beta0', value=twiss_init_beta)
            # specify the default Twiss
            doc.stag('default-range ref-name="defaultRange"')
    # specify the default Sequence
    doc.stag('default-sequence ref-name="' + default_sequence + '"')

    # Init files (i.e. actual sequence etc.)
    with tag('init-files'):
        for commonFile_ in commonInitialisationScripts:
            doc.stag('call-file', path = commonFile_)
    
    # Path to init: assuming XML is in the "operation" folder that is one level into repository
    with tag('path-offsets'):
        doc.stag('repository-prefix', value='../')
    
    # Standard field
    with tag('svn-revision'):
        text('$Revision$')

# Adjust indentation
result = indent(doc.getvalue(), indentation = ' '*2, newline = '\r\n')

with open(outputFilename, 'w') as f:
    print(result, file = f)
    
print('XML code for ' + machineName + ' written to file ' + outputFilename + '.')