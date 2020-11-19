"""
===========================
Murmann ADC survey analyzer
===========================

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
        self.print_log(type='I', msg='Initializing %s' %(__name__)) 

        self.db = {}
        self.plot = True
        self.export = (False,'')

        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;

        self.init()

    def init(self):
        try:
            self.download()
            self.extract_csv()
            self.process_csv()
        except:
            self.print_log(type='E',msg='Failed initializing ADC survey data.')

    @property
    def _classfile(self):
         return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__


    @property
    def revision(self):
        '''This should be eventually fetched form the website.

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
        '''Extract CSV files from the database database'''
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
        '''Process the CSV files to a dictionary.
        '''
        self.db = {}
        for key,val in self.databasefiles.items():
            firstrow = True
            reader = csv.reader(open(val, 'r'))
            self.db[key] = {}
            for row in reader:
                if firstrow:
                    firstrow = False
                    for title in row:
                        self.db[key][title] = []
                else:
                    keys = list(self.db[key].keys())
                    for i,k in enumerate(keys):
                        self.db[key][k].append(row[i])

    def _legend_without_duplicate_labels(self,ax):
        '''Adds legend with unique entries to the scatter plot.
        '''
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) == 0:
            return
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        fig = plt.gcf()
        try:
            fontsize = plt.rcParams['legend.fontsize']-2
        except:
            fontsize = 'small'
        ax.legend(*zip(*unique),loc=2,handlelength=1,fontsize=fontsize)

    def plot_fom(self,**kwargs):
        '''Plot an FoM scatter plot.

        Parameters
        ----------
        **kwargs :
            xdata : str, default 'fsnyq'
                Column header matching the x-axis data. This is matched to the
                start of the column header (case insensitive). For example,
                'fomw_hf' matches to 'FOMW_hf [fJ/conv-step]'.
            ydata : str, default 'fomw_hf'
                Column header matching the y-axis data. This is matched to the
                start of the column header (case insensitive).
            log : str, optional, default ''
                Set x- or y-axis to logarithmic scale. Possible values are
                'x','y' and 'xy'.
            cond : tuple or list(tuple), optional, default None
                Give conditions to filter out points from the scatter plot. The
                conditions are given as tuples with 3 elements each. The tuple
                is formed as (key,condition,value), where the key is matched to
                a column header in the same way as for xdata and ydata,
                condition is a string from {'<','<=','==','!=','>=','>'}, and
                value is a value in the same units as the column data. Multiple
                conditions can be given by wrapping the tuples in a list. If
                the condition value is a string, it is matched as
                'key.contains(value)' (case sensitive). 
            group : list(str), default None
                Manual grouping of ADC architectures. A group is created for
                each entry in the list. Architectures matching several groups
                are grouped into a separate group automatically (up to 2
                overlaps).  This functionality can be turned off with
                simplify_group.
            simplify_group : bool, default False
                Flag to simplify grouping based on entries in group. If True,
                the secondary groups, i.e. combinations of entries, are not
                separated.  In this case, the order of group entries defines
                the 'dominant' group. For example, 'SAR, TI' can be shown as
                just 'TI' or 'SAR' if both are enabled, depending on the order. 
            legend : bool, default True
                Flag to turn legend on or off. Legend entries include
                architectures filtered by either cond or group, and manually
                hilighted datapoints.
            datapoints : tuple or list(tuple), default None
                Hilighted datapoints to be added to the plot (not in the
                survey).  The tuple(s) should be pairs of (x,y), where the
                units of both x and y match the units of xdata and ydata. The
                datapoint can be labeled by including a third element in the
                tuple as (x,y,label).  Default label is 'This Work'.
            grayscale : bool, default False
                Flag to turn plot colors on or off. When grayscale is enabled,
                the ADC architectures are grouped by marker style rather than
                color.
            colormap : str, default 'jet'
                Colormap to be used for color picking. See matplotlib colormaps
                for options.

        '''
        xdata = kwargs.get('xdata','fsnyq')
        ydata = kwargs.get('ydata','fomw_hf')
        log = kwargs.get('log','')
        cond = kwargs.get('cond',None)
        group = kwargs.get('group',None)
        simplify_group = kwargs.get('simplify_group',False)
        legend = kwargs.get('legend',True)
        datapoints = kwargs.get('datapoints',None)
        grayscale = kwargs.get('grayscale',False)
        colormap = kwargs.get('colormap','jet')

        fig,ax = plt.subplots(constrained_layout=True)
        plt.grid(True)
        ax.set_axisbelow(True)
        if 'x' in log:
            plt.xscale('log')
        if 'y' in log:
            plt.yscale('log')
        markers = ['o','s','^','v','<','>','+','x','D','p','P','X','.']
        markerdict = {}
        ax.margins(x=0.05)
        tmpdict = copy.deepcopy(self.db.copy())
        archs = tmpdict['ISSCC']['ARCHITECTURE']+tmpdict['VLSI']['ARCHITECTURE']
        unique_arch = list(np.unique(archs))
        unique_arch.remove('')
        unique_arch = sorted(unique_arch)
        cmap = plt.cm.get_cmap(colormap,len(unique_arch))
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
            xvec,yvec = val[xkey],val[ykey]
            xvec = np.array([np.nan if x == '' else float(x) for x in xvec])
            yvec = np.array([np.nan if y == '' else float(y) for y in yvec])
            for arch in unique_arch:
                idcs = np.where(np.array(val['ARCHITECTURE'])==arch)[0]
                if len(idcs) == 0:
                    continue
                color = 'k'
                altcolor = None
                marker = '.'
                label = None
                fs = 'full'
                mew = 0
                ms = 1
                zorder = 1
                if group is not None:
                    # Manual grouping
                    nmatch = 0
                    for h in group:
                        if h in arch:
                            nmatch += 1
                            zorder = 1.5
                            if grayscale:
                                color = 'k'
                                mew = 1
                                marker = markers[group.index(h)]
                                fs = 'none'
                                if nmatch > 1:
                                    marker = markers[len(h)+group.index(h)+group.index(label)]
                            else:
                                ms = 1.2
                                print('----')
                                print(arch,color,altcolor)
                                if not simplify_group and nmatch > 1:
                                    altcolor = color
                                    fs = 'right'
                                color = cmap((group.index(h)+1)/(2*len(h)))
                                #color = cmap(unique_arch.index(arch)/len(unique_arch))
                                print(arch,color,altcolor)
                                marker = 'o'
                            if not simplify_group and nmatch > 1:
                                label += ', %s' % h
                            else:
                                label = h
                elif cond is not None:
                    # Grouping based on given conditions
                    label = arch
                    if grayscale:
                        if arch not in markerdict:
                            try:
                                markerdict[arch] = markers.pop(0)
                            except:
                                markerdict[arch] = '.'
                                self.print_log(type='W',msg='Ran out of markers. Filter architectures or use manual grouping.')
                        marker = markerdict[arch]
                        mew = 1
                        color = 'k'
                        fs = 'none'
                    else:
                        ms = 1.2
                        marker = 'o'
                        color = cmap(unique_arch.index(arch)/len(unique_arch))
                        fs = 'full'
                else:
                    # No grouping
                    marker = 'o'
                    if grayscale:
                        color = 'k'
                    else:
                        color = cmap(unique_arch.index(arch)/len(unique_arch))
                plt.plot(xvec[idcs],yvec[idcs],ls='none',marker=marker,\
                        fillstyle=fs,c=color,markerfacecoloralt=altcolor,markeredgewidth=mew,\
                        label=label,ms=plt.rcParams['lines.markersize']*ms,zorder=zorder)
        if datapoints is not None:
            if not isinstance(datapoints,list):
                datapoints = [datapoints]
            for d in datapoints:
                msize = plt.rcParams['lines.markersize']*1.5
                if len(d) > 2:
                    label = d[2]
                else:
                    label = 'This Work'
                plt.plot(d[0],d[1],ls='none',c='r',label=label,marker='*',ms=msize,\
                        markeredgewidth=0.5)
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
            fname = "%s_scatter.pdf"%self.export[1]
            fpath = os.path.dirname(fname) 
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            self.print_log(type='I',msg='Saving figure to %s.' % fname)
            plt.savefig(fname,format='pdf',bbox_inches='tight')
        if self.plot:
            plt.show(block=False)
        else:
            plt.close()

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from  murmann_adc_survey_analyzer import *
    import pdb
    # Example plot format settings, these are optional
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{mathptmx}']
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'normal'
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.grid.axis'] = 'both'
    plt.rcParams['grid.color'] = '0.75'
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize'] = 4
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 1
    plt.rcParams['legend.edgecolor'] = '0'
    plt.rcParams['patch.linewidth'] = '0.5'
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['figure.titlesize'] = 10
    plt.rcParams['figure.figsize'] = (3.5,2.0)
    plt.rcParams['figure.dpi'] = 150
    a=murmann_adc_survey_analyzer()
    cond = []
    cond.append(('fsnyq','>=',100e6))
    cond.append(('fsnyq','<=',5e9))
    cond.append(('fomw_hf','<=',1000))
    group = ['TI','SAR','VCO']
    gs = False
    a.export=(True,'../figures/1')
    a.plot_fom(xdata='fsnyq',log='xy',grayscale=gs)
    a.export=(True,'../figures/2')
    a.plot_fom(xdata='fsnyq',log='xy',group=group,grayscale=gs)
    a.export=(True,'../figures/3')
    a.plot_fom(xdata='fsnyq',log='xy',cond=cond,group=group,grayscale=gs)
    a.export=(True,'../figures/4')
    a.plot_fom(xdata='year',log='y',cond=cond,group=group,grayscale=gs)

    input()
