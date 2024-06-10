#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:25:53 2023

@author: albertsmith
"""

import os
from copy import copy


class CellReader():
    def __init__(self,filename:str):
        self.f=open(filename,'r')
        self.reset()
    
    def reset(self):
        self.f.seek(0)
        self.header=[]
        self._footer=None
        for line in self.f:
            self.header.append(line)
            if '"cells"' in line:break
        self.cont=True
        self.last_cell=[]
    
    def ReadCell(self):
        """
        Reads the next cell and returns all lines in that cell

        Returns
        -------
        out : TYPE
            DESCRIPTION.

        """
        f=self.f
        
        if not(self.cont):return None #End of cells already reached
        
        for line in f:
            if ']' in line:
                self.cont=False #Cell block is closed
                return None
            if '{' in line:
                break #New Cell has started
        
        out=[]
        count=1
        for line in f:
            count+=line.count('{')
            count-=line.count('}')
            # if '{' in line:count+=1
            # if '}' in line:count-=1
            if count<1:break
            out.append(line)
        self.last_cell=out
        
        
        return out

    @property
    def footer(self):
        if self._footer is None:
            self._footer=['],\n']
            while self.cont:self.ReadCell()
            for line in self.f:
                self._footer.append(line)
        return self._footer
        
        
    def get_source(self,lines:list=None):
        """
        Extracts the source information from the previously extracted cell or
        the provided lines

        Parameters
        ----------
        lines : list, optional
            List of lines to extract source from. The default is None.

        Returns
        -------
        list

        """
        if lines is None:lines=self.last_cell
        lines=copy(lines)
        
        while len(lines):
            line=lines.pop(0)
            if '"source"' in line:break
        count=1
        out=[]
        while len(lines):
            line=lines.pop(0)
            if '[' in line:count+=1
            if ']' in line:count-=1
            if count==0:break
            out.append(line)
        
        return out
        
    def __next__(self):
        if self.cont:
            return self.ReadCell()
        else:
            raise StopIteration
    
    def __iter__(self):
        self.reset()
        return self
    
    def __exit__(self):
        self.f.close()
        

def write_cell(f,lines:list):
    """
    Write lines to a cell for a file

    Parameters
    ----------
    f : file handle
        File to be written to
    lines : list
        Lines to go into the cell.

    Returns
    -------
    None.

    """

    f.write('{\n')
    for line in lines:
        f.write(line)
    f.write('}')


def write_colab_setup(f):
    """
    Append lines to code to set up Google colab with pyDR

    Parameters
    ----------
    f : file handle

    Returns
    -------
    None.

    """
    
    
    
    f.write("""{
     "cell_type": "code",
     "execution_count": 0,
     "id": "759eab0f",
     "metadata": {},
     "outputs": [],
     "source": [
      "# SETUP pyDR\\n",
      "!git clone https://github.com/alsinmr/pyDR.git"
     ]
    }""")
    
def write_book_setup(f):
    """
    Append lines to code to set up pyDR for r 

    Parameters
    ----------
    f : file handle

    Returns
    -------
    None.

    """
    
    
    
    f.write("""{
     "cell_type": "code",
     "execution_count": 0,
     "id": "759eab0f",
     "metadata": {},
     "outputs": [],
     "source": [
      "# SETUP pyDR\\n",
      "import os\\n",
      "os.chdir('../..')"
     ]
    }""")
    

def add_links(f,filename):
    """
    Adds links (Colab, download) to the notebook for the website

    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    f.write("""{
     "cell_type": "markdown",
     "id": "759eab0e",
     "metadata": {},
     "source": [""")
    f.write(f"""
      "<a href=\\"https://githubtocolab.com/alsinmr/pyDR_tutorial/blob/main/ColabNotebooks/{filename}\\" target=\\"_blank\\"><img src=\\"https://colab.research.google.com/assets/colab-badge.svg\\"></a>"
     """)
    # f.write(f"""
    #         "\\n\\n<a href=\\"https://github.com/alsinmr/pyDR_tutorial/raw/main/{filename}\\" target=\\"_blank\\"><img src=\\"Download-button.png\\" width=\\"100\\"></a>"
    #         """)
    f.write("""
            ]\n}""")
        
def copy2colab(filename):
    cr=CellReader(filename)
    first=True
    with open(os.path.join('ColabNotebooks',filename),'w') as f:
        for line in cr.header:
            f.write(line)
        for cell in cr:
            if cell is None:
                break
            elif len(cr.get_source()) and '(hidden on colab)' in cr.get_source()[0]:
                pass
            elif len(cr.get_source()) and 'SETUP pyDR' in cr.get_source()[0]:
                f.write('\n' if first else ',\n')
                write_colab_setup(f)
            else:
                f.write('\n' if first else ',\n')
                write_cell(f,cell)
            first=False
        f.write('\n')
        for line in cr.footer:
            f.write(line)
    cr.__exit__()
    
def copy2JupyterBook(filename):
    cr=CellReader(filename)
    first=True
    with open(os.path.join('JupyterBook',filename),'w') as f:
        for line in cr.header:
            f.write(line)
        for k,cell in enumerate(cr):
            if cell is None:
                break
            elif len(cr.get_source()) and '(hidden on webpage)' in cr.get_source()[0]:
                pass
            elif len(cr.get_source()) and 'SETUP pyDR' in cr.get_source()[0]:
                f.write('\n' if first else ',\n')
                write_book_setup(f)
            else:
                f.write('\n' if first else ',\n')
                write_cell(f,cell)
            if first:
                first=False
                f.write(',\n')
                add_links(f,filename)
                
        f.write('\n')
        for line in cr.footer:
            f.write(line)
    cr.__exit__()
    
  

if __name__=='__main__':
    from SetupNotebooks import *
    
    directory='/Users/albertsmith/Documents/GitHub/pyDR_tutorial'
    if not(os.path.exists(directory)):
        directory=directory.replace('GitHub','GitHub.nosync')
    
    #Load the file_records (modified times) if it exists
    file_record={}
    if os.path.exists('file_record.txt'):
        with open('file_record.txt','r') as f:
            for line in f:
                key,value=line.strip().split('\t')
                file_record[key]=int(value)
            
    with open('file_record.txt','w') as f:
        for filename in os.listdir(directory):
            if len(filename)>6 and filename[-6:]=='.ipynb':
                mt=int(os.path.getmtime(filename))
                if not(filename in file_record and mt==file_record[filename]):
                    #new or modified notebooks get copied
                    copy2colab(filename)
                    copy2JupyterBook(filename)
                
                f.write(f'{filename}\t{mt}\n')
        
        