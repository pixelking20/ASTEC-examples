###########################################################################
###########################################################################
## Copyright (C) 2018  Guignard Leo <guingardl__@__janelia.hhmi.org>     ##
##                                                                       ##
## This program is free software: you can redistribute it and/or modify  ##
## it under the terms of the GNU General Public License as published by  ##
## the Free Software Foundation, either version 3 of the License, or     ##
## (at your option) any later version.                                   ##
##                                                                       ##
## This program is distributed in the hope that it will be useful,       ##
## but WITHOUT ANY WARRANTY; without even the implied warranty of        ##
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         ##
## GNU General Public License for more details.                          ##
##                                                                       ##
## You should have received a copy of the GNU General Public License     ##
## along with this program.  If not, see <http://www.gnu.org/licenses/>. ##
###########################################################################
###########################################################################

import numpy as np
from matplotlib import pyplot as plt
import os, sys
import scipy.cluster.hierarchy as H
from copy import deepcopy
from scipy import ndimage as nd
import xml.etree.ElementTree as ET
if sys.version_info[0] < 3:
    import cPickle as pkl
else:
    import pickle as pkl

keydictionary = {'lineage': {'output_key': 'cell_lineage',
                             'input_keys': ['lineage_tree', 'lin_tree', 'Lineage tree', 'cell_lineage']},
                 'h_min': {'output_key': 'cell_h_min',
                           'input_keys': ['cell_h_min', 'h_mins_information']},
                 'volume': {'output_key': 'cell_volume',
                            'input_keys': ['cell_volume', 'volumes_information', 'volumes information', 'vol']},
                 'surface': {'output_key': 'cell_surface',
                             'input_keys': ['cell_surface', 'cell surface']},
                 'compactness': {'output_key': 'cell_compactness',
                                 'input_keys': ['cell_compactness', 'Cell Compactness', 'compacity',
                                                'cell_sphericity']},
                 'sigma': {'output_key': 'cell_sigma',
                           'input_keys': ['cell_sigma', 'sigmas_information', 'sigmas']},
                 'label_in_time': {'output_key': 'cell_labels_in_time',
                                   'input_keys': ['cell_labels_in_time', 'Cells labels in time', 'time_labels']},
                 'barycenter': {'output_key': 'cell_barycenter',
                                'input_keys': ['cell_barycenter', 'Barycenters', 'barycenters']},
                 'fate': {'output_key': 'cell_fate',
                          'input_keys': ['cell_fate', 'Fate']},
                 'fate2': {'output_key': 'cell_fate_2',
                           'input_keys': ['cell_fate_2', 'Fate2']},
                 'fate3': {'output_key': 'cell_fate_3',
                           'input_keys': ['cell_fate_3', 'Fate3']},
                 'fate4': {'output_key': 'cell_fate_4',
                           'input_keys': ['cell_fate_4', 'Fate4']},
                 'all-cells': {'output_key': 'all_cells',
                               'input_keys': ['all_cells', 'All Cells', 'All_Cells', 'all cells', 'tot_cells']},
                 'principal-value': {'output_key': 'cell_principal_values',
                                     'input_keys': ['cell_principal_values', 'Principal values']},
                 'name': {'output_key': 'cell_name',
                          'input_keys': ['cell_name', 'Names', 'names', 'cell_names']},
                 'contact': {'output_key': 'cell_contact_surface',
                             'input_keys': ['cell_contact_surface', 'cell_cell_contact_information']},
                 'history': {'output_key': 'cell_history',
                             'input_keys': ['cell_history', 'Cells history', 'cell_life', 'life']},
                 'principal-vector': {'output_key': 'cell_principal_vectors',
                                      'input_keys': ['cell_principal_vectors', 'Principal vectors']},
                 'name-score': {'output_key': 'cell_naming_score',
                                'input_keys': ['cell_naming_score', 'Scores', 'scores']},
                 'problems': {'output_key': 'problematic_cells',
                              'input_keys': ['problematic_cells']},
                 'unknown': {'output_key': 'unknown_key',
                             'input_keys': ['unknown_key']}}

def _set_dictionary_value(root):
    if len(root) == 0:
        if root.text is None:
            return None
        else:
            return eval(root.text)
    else:
        dictionary = {}
        for child in root:
            key = child.tag
            if child.tag == 'cell':
                key = int(child.attrib['cell-id'])
            dictionary[key] = _set_dictionary_value(child)
    return dictionary

def xml2dict(path):
    tree = ET.parse(path)
    root = tree.getroot()
    dictionary = {}

    for k, v in keydictionary.items():
        if root.tag == v['output_key']:
            dictionary[str(root.tag)] = _set_dictionary_value(root)
            break
    else:
        for child in root:
            value = _set_dictionary_value(child)
            if value is not None:
                dictionary[str(child.tag)] = value
    return dictionary


def read_property_file(path):
    ext = path.split(os.path.extsep)[-1]
    if ext == 'pkl':
        with open(path, 'rb') as f:
            if sys.version_info[0] < 3:
                properties = pkl.load(f)
            else:
                properties = pkl.load(f, encoding='latin1')
            f.close()
    else:
        properties = xml2dict(path)
    return properties

def get_ratio_mother_sisters_volume(mother, vals, lin_tree, inv_lin_tree, around=5):
    """ From a mother cell id (right before a division),
        computes the average value for a given attribute `vals` accros `around` timepoints
        for the mother cell and the daughter cells
        Return the ratio of the sum of the average of attribute 
        of the two sisters over the average of the attribute for the mother.
        Args:
            mother(int): id of the mother
            vals(dict): dictionary that maps a cell id to a numerical value
            around(int): time window to consider in time-points
        Returns:
            (int): The ration of the sum of the two sister cell values
                   over the mother cell value.
    """
    before = [mother]
    for i in range(around):
        before.append(inv_lin_tree.get(before[-1], before[-1]))
    after1 = [lin_tree[mother][0]]
    after2 = [lin_tree[mother][1]]
    for i in range(around):
        after1.append(lin_tree.get(after1[-1], [after1[-1]])[0])
        after2.append(lin_tree.get(after2[-1], [after2[-1]])[0])
    return ((np.mean([vals.get(c, 0) for c in after1]) + np.mean([vals.get(c, 0) for c in after2])) / 
            np.mean([vals.get(c, 0) for c in before]))

def get_symetric_volumes(couple, vals, lin_tree):
    """ Computes the average values of a given couple of cells
        throughout their life-span of the attribute `vals`
        Args:
            couple(list): list of two cells (usually symetrical cells but not necessary)
            vals(dict): dictionary that maps a cell id to a numerical value
        Returns:
            ((int), (int)): The average for the two cells of 
                            the given attribute throughout their life-spans
    """
    c1 = [couple[0]]
    c2 = [couple[1]]
    while len(lin_tree.get(c1[-1], []))==1 or len(lin_tree.get(c2[-1], []))==1:
        if len(lin_tree.get(c1[-1], []))==1:
            c1.append(lin_tree[c1[-1]][0])
        if len(lin_tree.get(c2[-1], []))==1:
            c2.append(lin_tree[c2[-1]][0])
    return np.mean([vals.get(c) for c in c1]), np.mean([vals.get(c) for c in c2])

def get_life_span(couple, lin_tree):
    """ Computes the lifespan of a couple of cells
        Args:
            couple(list): list of two cells (usually symetrical cells but not necessary)
        Returns:
            ((int), (int)): the lenght of the lifespan of the couple of cells
    """
    c1 = [couple[0]]
    c2 = [couple[1]]
    while len(lin_tree.get(c1[-1], []))==1 or len(lin_tree.get(c2[-1], []))==1:
        if len(lin_tree.get(c1[-1], []))==1:
            c1.append(lin_tree[c1[-1]][0])
        if len(lin_tree.get(c2[-1], []))==1:
            c2.append(lin_tree[c2[-1]][0])
    return [len(c1), len(c2)]


def get_values_around_division(c, vals, time_around, lin_tree, inv_lin_tree):
    """Compute the metric for a cell c around its division. 
    Before the division the metric used is the one of the mother,
    after the division it is the one of the clone of the mother.
    Args:
        c(int): cell id from the lineage tree
        vals(dict): dictionary that maps a cell id to a numerical value
        time_around(int): the time to consider before and after the division
    
    Returns:
        4 lists:
            The metric for the mother from *time_around* before division to the division`
            The metric for the daughter 1 from the division to *time_around* after it
            The metric for the daughter 2 from the division to *time_around* after it
    """
    mother_life = [c]
    for i in range(time_around):
        mother_life = [inv_lin_tree.get(mother_life[0], mother_life[0])] + mother_life
    sister_life = [lin_tree[c][0]]
    sister_life2 = [lin_tree[c][1]]
    for i in range(time_around):
        sister_life = sister_life + [lin_tree.get(sister_life[-1], [sister_life[-1]])[0]]
        sister_life2 = sister_life2 + [lin_tree.get(sister_life2[-1], [sister_life2[-1]])[0]]
    return ([vals.get(c, vals[lin_tree[c][0]]) for c in mother_life],
            [vals.get(c, None) for c in sister_life], 
            [vals.get(c, None) for c in sister_life2])

def plot_around_div(vals, lin_tree, inv_lin_tree, interest_cells, names, col=6, row=3, ylim=(), x_label='', y_label='', 
                    around=20, title='', saving_folder='', z_c_range = (8, 10), add_title='', legend_pos = 2, ncol = 2):
    """ For the cell cycles from 7 to 10, plots the evolution of a given attribute
        `vals` around the division events.
        Args:
            vals(dict): dictionary that maps a cell id to a numerical value
            col(int): #columns in the plot
            row(int): #rows in the plot
            ylim(tuple(int, int)): fixed ylim of the plot
            x_label(string): label of the x axis
            y_label(string): label of the y axis
            around(int): number of time-points around the divisions to consider
            title(string): title of the plot
            saving_folder(string): path to the directory where to save the plot
    """
    fig = plt.figure(figsize=(10, 8))
    i = 1
    for z_c in range(*z_c_range):
        ax = fig.add_subplot(row, col, i)
        whole = []
        whole_sis = []
        interest_cells_f = [c for c in interest_cells if int(names.get(c).split('.')[0][1:]) == z_c]
        for c in interest_cells_f:
            Yb, Yfs1, Yfs2 = get_values_around_division(c, vals, around, lin_tree, inv_lin_tree)
            Xb = list(range(-len(Yb)*2+2, 2, 2))
            Xe = list(range(0, max(len(Yfs1), len(Yfs2))*2, 2))
            Y = Yb
            Y1 = Yfs1
            ax.plot(Xe, Y1, '-', alpha=.2, color='k')
            ax.plot(Xb, Y, '-', alpha=.2, color='k')
            whole.append(list(Y))
            Y2 = Yfs2
            ax.plot(Xe, Y2, '-', alpha=.2, color='k')
            whole_sis.append(Y1)
            whole_sis.append(Y2)

        ax.set_xlim(2-len(Xb)*2, (len(Xe)-1)*2)            
        if i%row==1 or col==1:
            ax.set_ylabel(y_label, fontsize=30)
        else:
            ax.set_yticks([])
        if (i>=col*row-1 and col!=1) or (col==1 and i==row):
            ax.set_xlabel(x_label, fontsize=30)
        else:
            ax.set_xticks([])
        ax.text(.95,.9,'%dth zygotic division'%z_c,
            horizontalalignment='right',
            transform=ax.transAxes, fontsize=20,
            bbox={'facecolor':'white', 'pad':10, 'linewidth':3})
        if i==legend_pos:
            ax.plot([0, 0], [ylim[0], ylim[1]], 'k--', label='Moment of division')            
            whole = np.array(whole)
            ax.plot(Xb, np.median(whole, axis=0), 'r-', lw=3, label='Distribution median')
            ax.plot(Xb, np.percentile(whole, 10, axis=0), 'r--', lw=2, label='10% and 90%')
            ax.plot(Xb, np.percentile(whole, 90, axis=0), 'r--', lw=2)
            
            X=list(range(0, len(whole_sis[0])*2, 2))
            whole_sis = np.array(whole_sis)
            ax.plot(Xe, np.median(whole_sis, axis=0), 'r-', lw=3)
            ax.plot(Xe, np.percentile(whole_sis, 10, axis=0), 'r--', lw=2)
            ax.plot(Xe, np.percentile(whole_sis, 90, axis=0), 'r--', lw=2)
            ax.legend(fontsize=15, loc='lower left', ncol=ncol)
        else:
            ax.plot([0, 0], [ylim[0], ylim[1]], 'k--')      
            whole = np.array(whole)
            ax.plot(Xb, np.median(whole, axis=0), 'r-', lw=3)
            ax.plot(Xb, np.percentile(whole, 10, axis=0), 'r--', lw=2)
            ax.plot(Xb, np.percentile(whole, 90, axis=0), 'r--', lw=2)
            
            X=list(range(0, len(whole_sis[0])*2, 2))
            whole_sis = np.array(whole_sis)
            ax.plot(Xe, np.median(whole_sis, axis=0), 'r-', lw=3)
            ax.plot(Xe, np.percentile(whole_sis, 10, axis=0), 'r--', lw=2)
            ax.plot(Xe, np.percentile(whole_sis, 90, axis=0), 'r--', lw=2)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_ylim(ylim)

                
        i += 1
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    if saving_folder!='':
        newpath=saving_folder
        if not os.path.exists(newpath): os.makedirs(newpath)
        fig.savefig(newpath+y_label+add_title+'_around_division.pdf')


def get_x_y(distribution, bins, range=(0, 0.8), normed=False):
    """ From a distribution computes an histogram and 
        returns the x-y position of the bins
        Args:
            distribution(list): list of numerical values
            bins(int): number of bins to compute
            range(float, float): range of the histogram
            normed(bool): whether the histogram should be normed or not
        Returns:
            X(list): x positions of the bins
            Y(list): y positions of the bins
    """
    vals=np.histogram(distribution, bins=bins, range=range, normed=normed)
    X=np.array(vals[1])
    X=(X[1:]+X[:-1])/2.
    return X, np.array(vals[0])

def build_distance_matrix(tree_distances, size=64):
    """ Build a size x size matrix of distances between lineage trees
        Args:
            tree_distance(dict): dictionary that maps a couple of 
                                 cell ids onto their pairwise distance
            size(int): size of the squared matrix
        Returns:
            X(np.array): size x size np.array where X[i, j] is the distance between i and j
            corres(dict): a dictionary that maps a cell id onto an index in X
            corres_inv(dict): a dictionary that maps an index in X onto a cell id
    """
    corres = {}
    index = 0
    X = np.zeros((size, size))
    for c1, c2 in list(tree_distances.keys()):
        if c1<c2:
            if c1 not in corres:
                corres[c1] = index
                index += 1
            if c2 not in corres:
                corres[c2] = index
                index += 1

    corres_inv={v:k for k, v in list(corres.items())}
    for i in range(max(corres.values())+1):
        for j in range(i+1, max(corres.values())+1):
            c1, c2=corres_inv[i], corres_inv[j]
            if (c1, c2) in tree_distances:
                X[i, j]=tree_distances[(c1, c2)]
                X[j, i]=tree_distances[(c1, c2)]
            else:
                X[i, j]=tree_distances[(c2, c1)]
                X[j, i]=tree_distances[(c2, c1)]
    return X, corres, corres_inv

def format_name(n):
    """ Function that format a name n removing the unnecessary '0'
        while keeping its original lenght by adding spaces at the end
        Args:
            n(string): cell name formated as follow: <a/b>#.####<*/_>
        Returns:
            (string): the formated name
    """
    size = len(n)
    first_part = n.split('.')[0]+'.'
    second_part = str(int(n.split('.')[1][:-1]))
    last_part = n.split('.')[1][-1]
    all_name = first_part + second_part + last_part
    #print n, size - len(all_name)
    for i in range(size - len(all_name)):
        all_name+=' '
    return all_name
    
def perform_and_plot_dendrogram(D1, corres, corres_inv, f, ColorMap, names,
                                method='ward', saving_folder='', distance_name='',
                                color_threshold=None, name_format=True, prefix=''):
    """ Given a distance matrix `D1` performs a hierarchical linkage
        and plot the corresponding dendrogram
        Args:
            D1(np.array): a n x n symmetrical matrix
            corres(dict): a dictionary that maps a cell id onto an index in D1
            corres_inv(dict): a dictionary that maps an index in D1 onto a cell id
            f(dict): fate map dictionary
            saving_folder(string): path to the directory where to save the plot
            distance_name(string): distance name to include in the title
            color_threshold(float): threshold to cut the tree and assign different
                                    colors to different resulting clusters
    """
    if not ColorMap is None:
        CMapFates=ColorMap(list(set(f.values()))+['Undetermined'], 'rainbow')
    Y = H.linkage(D1, method=method)
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)
    tmp = H.dendrogram(Y, leaf_rotation=.5, color_threshold=color_threshold)
    labels = []
    for c in tmp['leaves']:
        n = ''
        for s in f.get(corres_inv[c], 'Undetermined').split(' '):
            n += s[0]
        if name_format:
            labels.append(format_name(names[corres_inv[c]])+' '+f.get(corres_inv[c], 'Undetermined'))
        else:
            labels.append(names[corres_inv[c]])
        
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=30)
    for i in plt.gca().get_xticklabels():
        if not ColorMap is None:
            i.set_color(CMapFates(i.get_text()[9:]))
        i.set_weight('bold')
        i.set_fontsize(30)
    # fig.subplots_adjust(left=0.04, bottom=0.6, right=0.96, top=0.96)
    ax.tick_params(axis='both', which='major', labelsize=22)
    fig.tight_layout()
    if saving_folder!='':
        fig.savefig(saving_folder+prefix+'dendogram.pdf')
        
def perform_and_plot_dendrogram_celegans(D1, corres, corres_inv, f, ColorMap, names,
                                method='ward', saving_folder='', distance_name='',
                                color_threshold=None, name_format=True, prefix=''):
    """ Given a distance matrix `D1` performs a hierarchical linkage
        and plot the corresponding dendrogram
        Args:
            D1(np.array): a n x n symmetrical matrix
            corres(dict): a dictionary that maps a cell id onto an index in D1
            corres_inv(dict): a dictionary that maps an index in D1 onto a cell id
            f(dict): fate map dictionary
            saving_folder(string): path to the directory where to save the plot
            distance_name(string): distance name to include in the title
            color_threshold(float): threshold to cut the tree and assign different
                                    colors to different resulting clusters
    """
    if not ColorMap is None:
        CMapFates=ColorMap(list(set(f.values()))+['Undeter'], 'Paired')
    Y = H.linkage(D1, method=method)
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)
    tmp = H.dendrogram(Y, leaf_rotation=.5, color_threshold=color_threshold)
    labels = []
    label_corres = {}
    for c in tmp['leaves']:
        n = ''
        for s in f.get(corres_inv[c], 'Undeter').split(' '):
            n += s[0]
        if name_format:
            labels.append((names[corres_inv[c]]))#+' '+f.get(corres_inv[c], 'Undeter'))
            label_corres[names[corres_inv[c]]] = f.get(corres_inv[c], 'Undeter')
        else:
            labels.append(names[corres_inv[c]])
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=30)
    for i in plt.gca().get_xticklabels():
        if not ColorMap is None:
            i.set_color(CMapFates(label_corres[i.get_text()]))                
        i.set_weight('bold')
        i.set_fontsize(30)
    fig.subplots_adjust(left=0.04, bottom=0.6, right=0.96, top=0.96)
    ax.tick_params(axis='both', which='major', labelsize=22)
    fig.tight_layout()
    if saving_folder!='':
        fig.savefig(saving_folder+prefix+'dendogram.pdf')

def get_min_vol(k, vals, lin_tree, size=10, decal=5):
    """ Given a cell id return the average attribute of its two daughter that is minimum
        (starting after `decal` time points for `size` time points)
        Args:
            k(int): a cell id that divide next time point
            vals(dict): dictionary that maps a cell id to a numerical value
            size(int): number of time point over which to average
            decal(int): number of time points not to consider after the division
        Returns:
            (float): the average attribute of its two daughter that is minimum
    """
    c1=lin_tree[k][0]
    c2=lin_tree[k][1]
    out=[]
    for i in range(decal):
        c1=lin_tree.get(c1, [c1])[0]
        c2=lin_tree.get(c2, [c2])[0]
    for i in range(size):
        out.append([vals[c1], vals[c2]])
        c1=lin_tree.get(c1, [c1])[0]
        c2=lin_tree.get(c2, [c2])[0]
    return min(np.mean(out, axis=0))

def get_max_vol(k, vals, lin_tree, size=10, decal=5):
    """ Given a cell id return the average attribute of its two daughter that is maximum
        (starting after `decal` time points for `size` time points)
        Args:
            k(int): a cell id that divide next time point
            vals(dict): dictionary that maps a cell id to a numerical value
            size(int): number of time point over which to average
            decal(int): number of time points not to consider after the division
        Returns:
            (float): the average attribute of its two daughter that is maximum
    """
    c1=lin_tree[k][0]
    c2=lin_tree[k][1]
    out=[]
    for i in range(decal):
        c1=lin_tree.get(c1, [c1])[0]
        c2=lin_tree.get(c2, [c2])[0]
    for i in range(size):
        out.append([vals[c1], vals[c2]])
        c1=lin_tree.get(c1, [c1])[0]
        c2=lin_tree.get(c2, [c2])[0]
    return max(np.mean(out, axis=0))

def short_name(n):
    """ Shorten a cell name
        Args:
            n(string): cell name
        Returns:
            (string): shorted cell name
    """
    return n.split('.')[0]+'.'+str(int(n.split('.')[1][:-1]))

def get_mother_name(n):
    """ From a cell name `n` gives the name that gave rise to this cell name
        Args:
            n(string): cell name
        Returns:
            (string): Name of the mother
    """
    letter=n[0]
    z_c=str(int(n.split('.')[0][1:])-1)
    num='%04d'%np.ceil(float(n.split('.')[1][:-1])/2)
    end=n[-1]
    return letter+z_c+'.'+num+end

def get_sister_name(n):
    """ From a cell name `n` gives the sister name
        Args:
            n(string): cell name
        Returns:
            (string): Name of the sister
    """
    return n.split('.')[0]+'.'+'%04d'%(int(n.split('.')[1][:-1])+1)+n[-1]

def generate_32_cell_stage(lin_tree, names, inv_lin_tree, sim_tree, sim_nv, vol):
    """ From a given lineage tree, names, lineage tree distances and volumes
        build a new lineage tree similar of the initial one with added the
        mother cells of the first cells of the initial lineage tree
        Args:
            lin_tree(dict): lineage tree
            names(dict): names dictionary
            inv_lin_tree(dict): inv_lin_tree dictionary
            sim_tree(dict): sim_tree dictionary
            sim_nv(dict): sim_nv dictionary
            vol(dict): vol dictionary
        Returns:
            new_lin_tree(dict): new lin_tree
            new_names(dict): new names
            new_vol(dict): new vol
            new_sim_nv(dict): new sim_nv
    """
    sim_tree = {(2*10**4+k[0], 2*10**4+k[1]):v for k, v in sim_tree}
    new_lin_tree = deepcopy(lin_tree)
    new_names = deepcopy(names)
    new_vol = deepcopy(vol)
    new_sim_nv = deepcopy(sim_nv)
    cells = [k for k in list(lin_tree.keys()) if k//10**4==1]
    i = 1
    for c in cells:
        if int(names[c].split('.')[1][:-1])%2==1:
            s_n = get_sister_name(names[c])
            for ci in cells:
                if names[ci] == s_n:
                    if c not in inv_lin_tree:
                        new_lin_tree[i] = [c, ci]
                    if (new_lin_tree[c][0], new_lin_tree[ci][0]) in sim_tree:
                        new_sim_nv[i] = sim_tree[(new_lin_tree[c][0], new_lin_tree[ci][0])]
                    else:
                        new_sim_nv[i] = sim_tree[(new_lin_tree[ci][0], new_lin_tree[c][0])]
                    new_names[i] = get_mother_name(names[c])
                    new_vol[i] = vol[c] + vol[ci]
                    i += 1
    return new_lin_tree, new_names, new_vol, new_sim_nv

# def compute_neighbors_stability(c, lin_tree, surf_ex, fates, surfaces, inv_lin_tree, 
#                                 th_presence = .1, th = .03, smoothing_percentage = 10):
#     """ From a lineage tree a dictionary of cell-cell contact area and
#         a dictionary of cell fates, computes the number of neighbors
#         that have changed for a given cell to the end of its cell cycle
#         Args:
#             c(int): id of the cell
#             lin_tree(dict): lineage tree
#             surf_ex(dict of dict): cell-cell contact area dictionary
#         Returns:

#     """
#     cell_fate = fates.get(c, "undeter")
#     cell_cycle = [c]
#     while len(lin_tree.get(cell_cycle[-1], [])) == 1:
#         cell_cycle += lin_tree[cell_cycle[-1]]

#     smoothing = int(np.ceil(len(cell_cycle) * float(smoothing_ratio)))
#     neighbors_correspondancy = {}
#     surface_evolution = {}
#     t = 0
#     for curr_c in cell_cycle:
#         surface_evolution[t] = {}
#         total_surf_evolution[t] = 0
#         for neighbor, s in surf_ex.get(curr_c, {}).iteritems():
#             neighbors_correspondancy[neighbor] = neighbors_correspondancy.get(inv_lin_tree.get(neighbor), neighbor)
#             to_treat = neighbors_correspondancy[neighbor]
#             if fates.get(to_treat, 'undeter') == cell_fate and s/surfaces[curr_c]>th:
#                 surface_evolution[t][to_treat] = surface_evolution[t].get(to_treat, 0) + s
#         t += 1

#     neighbs = np.array([vi for v in surface_evolution.values() for vi in v.keys()])
#     presence = nd.sum(np.ones_like(neighbs), neighbs, np.unique(neighbs))/len(cell_cycle)
#     nb_lost = np.sum((th_presence < presence) & (presence < 1 - th_presence))
#     return np.sum(th_presence < presence), float(nb_lost)/max(1, np.sum(th_presence < presence))

def compute_neighbors_stability(c, lin_tree, surf_ex, fates, surfaces, inv_lin_tree, 
                                th_presence = .1, th = .03, smoothing_percentage = 10):
    """ From a lineage tree a dictionary of cell-cell contact area and
        a dictionary of cell fates, computes the number of neighbors
        that have changed for a given cell to the end of its cell cycle
        Args:
            c(int): id of the cell
            lin_tree(dict): lineage tree
            surf_ex(dict of dict): cell-cell contact area dictionary
        Returns:

    """
    cell_fate = fates.get(c[0], "undeter")
    cell_cycle = c
#     while len(lin_tree.get(cell_cycle[-1], [])) == 1:
#         cell_cycle += lin_tree[cell_cycle[-1]]
    total_surf_evolution = {}
    smoothing = int(np.ceil(len(cell_cycle) * float(smoothing_percentage)))
    neighbors_correspondancy = {}
    surface_evolution = {}
    t = 0
    for curr_c in cell_cycle:
        surface_evolution[t] = {}
        total_surf_evolution[t] = 0
        for neighbor, s in list(surf_ex.get(curr_c, {}).items()):
            neighbors_correspondancy[neighbor] = neighbors_correspondancy.get(inv_lin_tree.get(neighbor), neighbor)
            to_treat = neighbors_correspondancy[neighbor]
            if fates.get(to_treat, 'undeter') == cell_fate and s/surfaces[curr_c]>th:
                surface_evolution[t][to_treat] = surface_evolution[t].get(to_treat, 0) + s
        t += 1

    neighbs = np.array([vi for v in list(surface_evolution.values()) for vi in list(v.keys())])
    presence = nd.sum(np.ones_like(neighbs), neighbs, np.unique(neighbs))/len(cell_cycle)
    nb_lost = np.sum((th_presence < presence) & (presence < 1 - th_presence))
    return np.sum(th_presence < presence), float(nb_lost)/max(1, np.sum(th_presence < presence))

def get_mother_names(n):
    letter = n[0]
    cycle = int(n.split('.')[0][1:])
    id_ = int(n.split('.')[1][:-1])
    return '%s%d.%d'%(letter, cycle-1, (id_+1)//2)

def get_daughters(n):
    l = n[0]
    cycle = int(n.split('.')[0][1:])
    id_ = int(n.split('.')[1])
    return '%s%d.%d'%(l, cycle+1, id_*2-1), '%s%d.%d'%(l, cycle+1, id_*2)

def name_comparison(c1, c2):
    cells_to_find_induction = ['a6.2', 'a6.3', 'a6.4', 'a6.7', 'b6.2', 'b6.3', 'b6.4',
                           'a7.8', 'a7.9', 'a7.10', 'a7.13', 'b7.3', 'b7.9', 'b7.10',
                           'a8.7', 'a8.8', 'a8.17', 'a8.19', 'a8.25',
                           'a8.15', 'a8.16']
    if c1 in cells_to_find_induction and not c2 in cells_to_find_induction:
        return -1
    if not c1 in cells_to_find_induction and c2 in cells_to_find_induction:
        return 1
    if int(c1[1]) < int(c2[1]):
        return -1
    if int(c2[1]) < int(c1[1]):
        return 1
    if c1[0] < c2[0]:
        return -1
    if c2[0] < c1[0]:
        return 1
    if int(c1.split('.')[-1]) < int(c2.split('.')[-1]):
        return -1
    else:
        return 1
