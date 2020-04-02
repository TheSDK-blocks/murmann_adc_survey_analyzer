"""
========
Murmann ADC survey analyzer
========

This entity provides an analyzer for Boris Murman ADCSyrvey at

https://web.stanford.edu/~murmann/publications/ADCsurvey_rev20200401.xls

Current docstring documentation style is Numpy
https://numpydoc.readthedocs.io/en/latest/format.html

This text here is to remind you that documentation is iportant.
However, youu may find it out the even the documentation of this 
entity may be outdated and incomplete. Regardless of that, every day 
and in every way we are getting better and better :).

Initially written by Marko Kosunen, marko.kosunen@aalto.fi, 2017.

"""

import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

import subprocess

from thesdk import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class murmann_adc_survey_analyzer(thesdk):
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.print_log(type='I', msg='Inititalizing %s' %(__name__)) 

        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;

        self.init()

    def init(self):
        pass #Currently nohing to add

    @property
    def _classfile(self):
         return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__


    @property
    def revision(self):
        '''This should be eventually fetched form the vwebsite.

        '''
        self._revision='20200401'
        return self._revision

    @property
    def databasefiles(self): 
        self._databasefiles={ key : self.entitypath+'/database/'
                +self.revision + '_' + key +'.csv' for key in [ 'ISSCC', 'VLSI'  ] }
        return self._databasefiles

    def download(self):
        '''Downloads the case database'''
        xlsfile=self.entitypath+ '/database/ADC_survey_rev' +self.revision+'.xls'
        if not os.path.exists(xlsfile):
            command= ('wget "https://web.stanford.edu/~murmann/publications/ADCsurvey_rev'
               + self.revision+'.xls" -O ' + xlsfile )
            self.print_log(type='I', msg='Executing %s \n' %(command))
            subprocess.check_output(command, shell=True);

    def extract_csv(self):
        '''Extract CSV files frm the database database'''
        
        for key,value in self.databasefiles.items():
            if key is 'ISSCC':
                sheet=2
            elif key is 'VLSI':
                sheet=3

            command=('ssconvert -S '+ self.entitypath + '/database/ADC_survey_rev'+self.revision 
                    + '.xls ' +
                    self.entitypath + '/database/tmpfile.csv  && mv '
                    + self.entitypath + '/database/tmpfile.csv.%s ' %(sheet) + value )
            subprocess.check_output(command, shell=True);
        command=('rm -f self.entitypath' + '/database/tmpfile.csv*')


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from  murmann_adc_survey_analyzer import *
    import pdb
    a=murmann_adc_survey_analyzer()
    a.download()
    a.extract_csv()

    #input()
