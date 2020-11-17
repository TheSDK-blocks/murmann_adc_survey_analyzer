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
import csv
import pdb
import copy

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

        self.db = {}
        self.plot = True
        self.export = (False,'')

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
                sheet=1
            elif key is 'VLSI':
                sheet=2
            command=('ssconvert -S '+ self.entitypath + '/database/ADC_survey_rev'+self.revision 
                    + '.xls ' +
                    self.entitypath + '/database/tmpfile.csv  && mv '
                    + self.entitypath + '/database/tmpfile.csv.%s ' %(sheet) + value )
            subprocess.check_output(command, shell=True);
        command=('rm -f ' + self.entitypath + '/database/tmpfile.csv*')
        subprocess.check_output(command, shell=True);

    def process_csv(self):
        self.db = {}
        for key,val in self.databasefiles.items():
            header = True
            reader = csv.reader(open(val, 'r'))
            self.db[key] = {}
            for row in reader:
                if header:
                    header = False
                    for title in row:
                        self.db[key][title] = []
                else:
                    keys = list(self.db[key].keys())
                    for i,k in enumerate(keys):
                        self.db[key][k].append(row[i])

    def _legend_without_duplicate_labels(self,ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        fig = plt.gcf()
        ax.legend(*zip(*unique),loc=2,frameon=True,borderpad=0.15,labelspacing=0.3,handletextpad=0.1,\
                borderaxespad=0.2,handlelength=1,framealpha=0.6,edgecolor='w',\
                fontsize=plt.rcParams['legend.fontsize']-1)

    def plot_fom(self,xdata='fsnyq',ydata='fomw_hf',log='',cond=None,legend=True,datapoints=None,label=None):
        from matplotlib.axes._axes import _log as matplotlib_axes_logger
        matplotlib_axes_logger.setLevel('ERROR')
        if legend:
            fig,ax = plt.subplots(constrained_layout=False)
            plt.tight_layout()
        else:
            fig,ax = plt.subplots()
        if 'x' in log:
            plt.xscale('log')
            plt.grid(True,which='both',axis='x')
        if 'y' in log:
            plt.yscale('log')
        ax.margins(x=0.05)
        tmpdict = copy.deepcopy(self.db.copy())
        archs = tmpdict['ISSCC']['ARCHITECTURE']+tmpdict['VLSI']['ARCHITECTURE']
        unique_arch = list(np.unique(archs))
        unique_arch.remove('')
        cmap = plt.cm.get_cmap('jet',len(unique_arch))
        for key,val in tmpdict.items():
            # Remove entries not matching conditions
            if cond is not None:
                if not isinstance(cond,list):
                    cond = [cond]
                for c in cond:
                    ckey = ''
                    for k in val.keys():
                        if k.lower().startswith(c[0].lower()):
                            ckey = k
                            break
                    if ckey in val:
                        isstr = False
                        try:
                            values = [float(x) if x != '' else np.nan for x in val[ckey]]
                        except:
                            isstr = True
                            values = [str(x) for x in val[ckey]]
                        if c[1] == '<':
                            mask = [e<c[2] if ~np.isnan(e) else False for e in values]
                        elif c[1] == '<=': 
                            mask = [e<=c[2] if ~np.isnan(e) else False for e in values]
                        elif c[1] == '==':
                            if isstr:
                                mask = [c[2] in e for e in values]
                            else:
                                mask = [e==c[2] if ~np.isnan(e) else False for e in values]
                        elif c[1] == '!=':
                            if isstr:
                                mask = [c[2] not in e for e in values]
                            else:
                                mask = [e!=c[2] if ~np.isnan(e) else False for e in values]
                        elif c[1] == '>=':
                            mask = [e>=c[2] if ~np.isnan(e) else False for e in values]
                        elif c[1] == '>':
                            mask = [e>c[2] if ~np.isnan(e) else False for e in values]
                        else:
                            mask = []
                        for k in val.keys():
                            for i,v in enumerate(val[k]):
                                if not mask[i]:
                                    val[k][i] = ''
            for k in val.keys():
                if k.lower().startswith(xdata.lower()):
                    xkey = k
                    break
            for k in val.keys():
                if k.lower().startswith(ydata.lower()):
                    ykey = k
                    break
            xvec = val[xkey]
            yvec = val[ykey]
            xvec = [np.nan if x == '' else float(x) for x in xvec]
            yvec = [np.nan if y == '' else float(y) for y in yvec]
            labelvec = []
            colorvec = []
            for i in range(len(xvec)):
                x = xvec[i]
                y = yvec[i]
                arch = val['ARCHITECTURE'][i]
                if arch == '':
                    continue
                # Fix in database?
                if arch == 'SAR TI':
                    arch = 'SAR, TI'
                color = cmap(unique_arch.index(arch)/len(unique_arch))
                labelvec.append(arch)
                colorvec.append(color)
                marker = 'o' if key == 'ISSCC' else 's'
                if not legend:
                    plt.scatter(x,y,c='k',label=arch,marker='o')
                else:
                    plt.scatter(x,y,c=color,label=arch,marker='o')
        if datapoints is not None:
            if not isinstance(datapoints,list):
                datapoints = [datapoints]
            for d in datapoints:
                msize = (plt.rcParams['lines.markersize']*2)**2
                plt.scatter(d[0],d[1],c='r',label=label,marker='*',s=msize)
        if legend:
            self._legend_without_duplicate_labels(ax)
        if plt.rcParams['text.usetex']:
            plt.xlabel(xkey.replace('_','\_'))
            plt.ylabel(ykey.replace('_','\_'))
        else:
            plt.xlabel(xkey)
            plt.ylabel(ykey)
        if 'year' in xkey.lower():
            from matplotlib.ticker import MaxNLocator
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xticks(rotation=30)
            plt.setp(ax.get_xticklabels(), ha="right")
        if self.export[0]:
            plt.savefig("%s_scatter.pdf"%self.export[1],format='pdf',bbox_inches='tight')
        if self.plot:
            plt.show(block=False)
        else:
            plt.close()

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from  murmann_adc_survey_analyzer import *
    import pdb
    a=murmann_adc_survey_analyzer()
    a.download()
    a.extract_csv()

    #input()
